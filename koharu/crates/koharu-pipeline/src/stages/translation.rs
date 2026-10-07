use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use koharu_scene::{Authored, EntityId, LanguageTag, Origin, SourceText, Translation};
use koharu_translator::{GenerationConfig, TranslationRequest, Translator};

use crate::{Progress, TranslationConfig};

use super::{ModelRef, ModelState, StageInput, StageProcessor, finish, generation};

const PRODUCER: &str = "dev.koharu.pipeline.translation";

pub(super) struct Processor {
    config: TranslationConfig,
    translator: Translator,
    last_used: AtomicU64,
    /// Whether the site's own text is removed before a region is translated.
    ///
    /// **Scoping the VERDICT is not enough, and the render proved it.** With only
    /// the verdict scoped, a display column carrying three names and a site's
    /// plate stopped being refused and then went to the translator whole -- so
    /// the page lettered
    /// `BLUE FLAME KING • FROST KING • GALE KING LATEST FREE MANGA WWW.PAPERLEAF.COM`
    /// down its own axis while the plate it came from was still on the artwork
    /// underneath. The three names are right; the tail is the site's, twice.
    ///
    /// Only the TARGET is stripped. `SourceText` stays whole, because `sfx.rs`,
    /// `duplicate.rs` and the extension's per-host latch all key on it, and the
    /// wire keeps reporting what was actually read.
    scope_watermark_refusals: bool,

    /// Whether a read the pipeline already refuses to letter is kept out of the
    /// request -- `ProcessorConfig::skip_unlettered_reads`.
    ///
    /// Snapshotted here rather than read from `StageInput`, which carries no
    /// config at all, and threaded into `targets` as a parameter for the reason
    /// `scope_watermark_refusals` already is: `has_work` and `process` must walk
    /// the SAME list, and a divergence between them is silent on every page it
    /// breaks.
    skip_unlettered_reads: bool,
}

impl Processor {
    pub(super) fn new(
        config: TranslationConfig,
        translator: Translator,
        scope_watermark_refusals: bool,
        skip_unlettered_reads: bool,
    ) -> Self {
        Self {
            config,
            translator,
            last_used: AtomicU64::new(0),
            scope_watermark_refusals,
            skip_unlettered_reads,
        }
    }
}

impl Processor {
    /// The exact request `process` sends, built in one place so a test can
    /// hold the object the provider will see.
    ///
    /// Extracted for the unwired-fix trap: a config lever that never reaches the
    /// request leaves every prompt-side unit test green while the shipping arm
    /// runs the baseline. A test on this method asserts the COMPOSED request
    /// -- the predicate the caller calls -- not the halves.
    fn build_request(&self, targets: &[Target], input: &StageInput) -> TranslationRequest {
        let mut request = TranslationRequest::new(
            targets.iter().map(|(_, source, _)| source.clone()),
            self.config.target_language,
        );
        if let Some(instructions) = self.config.instructions.as_deref() {
            request = request.with_instructions(instructions);
        }
        /* Snapshotted config, like `instructions`: the clause is a process-wide
         * arm, never a per-page judgement. The sentence itself and the
         * reasoning live in `prompt.rs`. */
        request = request.with_containment_clause(self.config.containment_clause);
        /* Per-request, unlike `instructions`, which is snapshotted config. This
         * is what lets a caller carry a story's earlier pages into this page's
         * prompt without a pipeline reload -- see `Request::context`. */
        let context = input.translation_context();
        if !context.is_empty() {
            request = request.with_context(context.iter().cloned());
        }
        // Same per-request carriage as context; law rather than precedent.
        let glossary = input.translation_glossary();
        if !glossary.is_empty() {
            request = request.with_glossary(glossary.iter().cloned());
        }
        /* `--segment-context`. Gathered unconditionally above because
         * it costs one component read per layer, and SPENT here or not at all --
         * with the flag off the vec stays empty, every segment serializes as the
         * bare `{id, text}` it always did, and the conditional prompt sentence
         * in `prompt.rs` does not fire. That is the byte-exact off arm, and
         * `the_off_arm_carries_no_segment_context` asserts it. */
        if self.config.segment_context {
            request.segment_context = targets
                .iter()
                .map(|(_, _, context)| context.clone())
                .collect();
        }
        request
    }

    /// The exact generation options `process` sends, config and per-request
    /// seed composed in one place for `build_request`'s reason above: a seed
    /// that never reached the call site would leave every parse-side test
    /// green while the retry re-rolled nothing. `None` -- every caller but
    /// the retry button -- leaves the config untouched, seed included, which
    /// is the fixed upstream constant and the byte-identical replay that
    /// measurements depend on. Per-request rather than `PipelineConfig` because a
    /// re-roll that forced a reload would pay a cold start to change one
    /// sample.
    fn generation_for(&self, input: &StageInput) -> GenerationConfig {
        let mut generation = self.config.generation;
        if let Some(seed) = input.translation_seed() {
            generation.seed = Some(seed);
        }
        generation
    }
}

impl ModelState for Processor {
    fn loaded(&self) -> bool {
        self.translator.loaded(&self.config.model)
    }

    fn unload(&self) -> bool {
        self.translator.unload()
    }

    fn touch(&self, sequence: u64) {
        self.last_used.store(sequence, Ordering::Relaxed);
    }

    fn last_used(&self) -> u64 {
        self.last_used.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> ModelRef<'_> {
        ModelRef::new(Translator::model(&self.config.model), self)
    }

    /// Nothing to translate means nothing to load, and this is the only place
    /// that can say so before the weights move.
    ///
    /// The check the pipeline had already existed -- `Translator::translate`
    /// returns early on an empty segment list -- but it sits one call *after*
    /// `stage_runner` has paged the model in, so a page with no text still paid
    /// a full cold load of the local LLM to be handed nothing.
    ///
    /// Deliberately the same `targets` walk `process` runs, and not a cheaper
    /// re-statement of it. The strings it allocates are a page's worth of short
    /// source lines; the load it can skip is 16.5 GiB. A second definition that
    /// could drift from the first is the only way this can go wrong, so there
    /// is no second definition.
    fn has_work(&self, input: &StageInput) -> Result<bool> {
        Ok(!targets(input, self.scope_watermark_refusals, self.skip_unlettered_reads)?.is_empty())
    }

    async fn load(&self) -> Result<()> {
        self.translator.load_model(&self.config.model).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let targets = targets(&input, self.scope_watermark_refusals, self.skip_unlettered_reads)?;
        let request = self.build_request(&targets, &input);
        let (provider, translated) = self
            .translator
            .translate(&self.config.model, self.generation_for(&input), request)
            .await?;

        // Reported before the patch is built, and never turned into an error.
        // Every segment the provider skipped is about to be written back as a
        // `Translation` holding the source text verbatim, which renders as
        // untranslated Japanese that nothing downstream -- not the patch, not
        // the report, not the scene -- can tell apart from a translation. This
        // is the only place that knows.
        //
        // Truncation is reported on its own, with an empty entity list. A reply
        // cut off *inside* the last segment's text is repaired by
        // `json::from_str` into a complete entry, so every id is answered and
        // `untranslated` is empty -- while that bubble ends mid-word. Landing
        // inside the final item is about as likely as landing before it, so
        // gating this on the entity list would report half of all overflows as
        // a clean page.
        //
        // A mis-addressed reply is reported on its own too, and it is the third
        // shape this gate has to admit. Under the local grammar a duplicate
        // always steals an id from somebody, so it arrives with a non-empty
        // entity list -- but a provider decoding unconstrained can answer more
        // entries than it was asked for, cover every id, and still have named one
        // of them twice. Nothing is wrong with that page, and the counter is
        // still the only evidence that "copy every input ID exactly once" is not
        // being followed.
        if translated.truncated
            || !translated.untranslated.is_empty()
            || translated.duplicate_ids > 0
            || translated.out_of_range_ids > 0
            // A cut page reports even when every id was answered: otherwise
            // repaired and silently-cut pages are indistinguishable from outside.
            || translated.cut_found > 0
        {
            let entities = translated
                .untranslated
                .iter()
                .filter_map(|&index| targets.get(index).map(|(entity, _, _)| *entity))
                .collect::<Vec<_>>();
            tracing::warn!(
                provider,
                page = %input.page(),
                untranslated = translated.untranslated.len(),
                segments = targets.len(),
                truncated = translated.truncated,
                duplicate_ids = translated.duplicate_ids,
                out_of_range_ids = translated.out_of_range_ids,
                // Generic on purpose: this now fires for a page with no missed
                // ids at all, where nothing is left in the source language and
                // the damage is a single bubble cut short -- or none at all, and
                // the only fault is an id answered twice. The fields say which.
                "translation returned an incomplete reply"
            );
            input.report(Progress::Untranslated {
                page: input.page(),
                entities,
                segments: targets.len(),
                truncated: translated.truncated,
                duplicate_ids: translated.duplicate_ids,
                out_of_range_ids: translated.out_of_range_ids,
                cut_found: translated.cut_found,
                still_cut: translated.still_cut,
            });
        }

        let language = LanguageTag::new(self.config.target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let mut edit = input.scene.edit_as(generated.clone());
        for (entity, _, _) in &targets {
            edit.observe::<SourceText>(*entity)?;
            edit.observe::<Translation>(*entity)?;
        }
        for ((entity, source, _), text) in targets.into_iter().zip(translated.segments) {
            if input
                .scene
                .component::<Translation>(entity)?
                .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            let text = if source.trim() == "\u{2026}" {
                "\u{2026}".to_owned()
            } else {
                text
            };
            edit.set(
                entity,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
            )?;
        }
        finish(edit)
    }
}

/// The segments this page would send to the model, paired with the content
/// entity each answer is written back to, in request order.
///
/// Shared by `process` and [`Processor::has_work`] so the two cannot disagree
/// about what "something to translate" means. That is not tidiness. `has_work`
/// answering `false` skips the stage outright, and a segment that never reaches
/// the model keeps its `SourceText` and renders as Japanese under an ordinary
/// `Translation` -- indistinguishable, downstream, from a page that genuinely
/// had nothing on it. A divergence here would be silent on every page it broke.
/// One translatable text, with what the pipeline knows ABOUT it.
///
/// The context travels in the same tuple as the text, and that is deliberate:
/// `segment_context` is positional, so any path that could produce the two lists
/// with different lengths or different filtering would silently label segments
/// with their neighbours' facts. Gathering both in one pass makes that
/// impossible rather than merely unlikely.
type Target = (EntityId, String, koharu_translator::SegmentContext);

/// The plain-English name for a segment, from its region kind AND its role.
///
/// The model reads prose, not our namespaces: `dev.koharu.region.onomatopoeia`
/// means nothing to it and "sound effect" means what it says. Unmapped
/// combinations return `None` and the field is then omitted entirely rather than
/// shipping a raw identifier, or a wrong word, into the prompt.
///
/// ## Why the ROLE is consulted and not just the kind
///
/// Every readable text on a page is `dev.koharu.region.text`, so kind alone
/// called a technique name and a line of speech the same thing -- "dialogue or
/// caption" -- and the resulting defect was transliterated ability names: on a
/// Chinese test chapter one ability name came back as pinyin with the label on
/// and as an English rendering with it off, twice, through the seam. Telling a
/// model that a title card is a caption is telling
/// it to preserve rather than translate, so the label was working against the
/// no-transliteration rule in the same prompt.
///
/// The distinction already exists and costs nothing to read: `link_dialogue_regions`
/// writes `dev.koharu.text.dialogue` on a text a detected BUBBLE contains
/// (`detection.rs:1806`) and leaves `dev.koharu.text.free-text` on everything
/// else (`:1586`). A balloon holds speech; free-standing display text on a
/// manhua action page is a title, a caption, or the name of a technique.
///
/// ## Why an unknown role is omitted rather than guessed
///
/// The wrong word is worse than no word -- that is this function's whole
/// lesson. A role we do not recognise yields `None`, the field disappears, and
/// the model is left to read the characters, which is what it did before any of
/// this existed.
fn segment_kind(kind: &str, role: Option<&str>) -> Option<&'static str> {
    match (kind, role) {
        ("dev.koharu.region.onomatopoeia", _) => Some("sound effect"),
        ("dev.koharu.region.text", Some("dev.koharu.text.dialogue")) => {
            Some("dialogue spoken in a speech balloon")
        }
        ("dev.koharu.region.text", Some("dev.koharu.text.free-text")) => {
            Some("display text: a title, a caption, or the name of a technique or ability")
        }
        _ => None,
    }
}

fn targets(input: &StageInput, scoped: bool, skip_unlettered: bool) -> Result<Vec<Target>> {
    let mut targets = Vec::new();
    if let Some(group) = input.scene.page(input.page)?.text_group()? {
        for layer in group.text_layers()? {
            if !input.contains_entity(layer.id())? {
                continue;
            }
            let content = layer.content()?;
            let Some(source) = content.source()? else {
                continue;
            };
            /* The OCR stage's own lettering veto, committed with that stage's
             * patch before this stage's input was built -- so it is readable
             * here, and it is the same fact the server reports as
             * `HIDDEN_LAYER_REFUSAL`. Asked of the SOURCE text below, never the
             * stripped `value`, because `hide_implausible` classifies
             * `source.text.value` too: if the two asked different questions a
             * region could be dropped here and still lettered there. */
            let hidden = layer
                .visibility()?
                .is_some_and(|visibility| !visibility.visible);
            let unlettered = super::unlettered_read(&source.text.value, hidden, scoped);
            // The strip happens HERE and nowhere later, because there is no
            // afterwards to do it in: source and translated line counts do not
            // correspond, so no post-hoc index mapping could remove the site's
            // half of a reply. Furniture that reaches the model comes back
            // translated and is lettered.
            let value = if scoped {
                let stripped = super::story_text(&source.text.value);
                // Empty means the region is nothing but furniture. It is already
                // refused by the gates; pass it through unchanged so `has_work`
                // and `nothing_translated` behave exactly as they do today.
                if stripped.is_empty() {
                    source.text.value
                } else {
                    stripped
                }
            } else {
                source.text.value
            };
            if !value.trim().is_empty() {
                /* Read off the SAME `layer`/`content` pair the text came from,
                 * inside the same guard, so a segment and its context cannot
                 * come from different regions. */
                let struck = layer.typography()?.is_some_and(|typography| {
                    typography
                        .extensions
                        .contains_key(koharu_scene::STRIKE_COLOR_EXTENSION)
                });
                let role = content.role()?;
                let kind = content
                    .source_region()?
                    .map(|region| region.region())
                    .transpose()?
                    .and_then(|region| {
                        segment_kind(region.kind.as_str(), role.as_ref().map(|r| r.role.as_str()))
                    })
                    .map(str::to_owned);
                targets.push((
                    (
                        content.id(),
                        value,
                        koharu_translator::SegmentContext { kind, struck },
                    ),
                    unlettered,
                ));
            }
        }
    }
    /* THE FILTER IS NEVER ALLOWED TO EMPTY A PAGE, and the guard is the whole
     * safety argument rather than a tidy-up. `has_work` is exactly
     * `!targets(..).is_empty()`, and a false there SKIPS the stage: `process`
     * never runs, so `Progress::Untranslated` never fires, so `Misses` stays at
     * its default and `nothing_translated`'s 502 becomes structurally
     * unreachable -- while the renderer's `fallback_to_source_text`, which
     * defaults ON, paints the raw source in a translation font. A page that
     * shouted would go quiet and ship untranslated instead.
     *
     * So the survivor test below makes `has_work` provably identical in both
     * arms: a page with at least one lettered read keeps at least one target,
     * and a page with none keeps every target it has today. The pages this
     * gives up on -- 62 of 173 on the measured chapter, where EVERY region is
     * refused -- cost only time: their regions are all refused downstream, and
     * `routes.rs:1424` already keeps a refused pair out of the story window, so
     * they perturb nothing further even when translated. */
    let filtered = skip_unlettered && targets.iter().any(|(_, unlettered)| !unlettered);
    Ok(targets
        .into_iter()
        .filter(|(_, unlettered)| !(filtered && *unlettered))
        .map(|(target, _)| target)
        .collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use koharu_scene::{At, EntityId, PageDraft, Session, Snapshot, TextLayout, TextLayoutKind};

    use crate::{ImageCache, TranslationConfig};

    use super::{Processor, StageInput, StageProcessor, segment_kind, targets};

    /// AN ABILITY TITLE IS NOT A CAPTION, which is the whole of the split.
    ///
    /// Kind alone called a technique name and a line of speech the same thing,
    /// and the defect it caused was transliterated ability names. Telling a
    /// model that a title card is a caption tells it
    /// to preserve rather than translate, which worked against the
    /// no-transliteration rule sitting in the same prompt.
    ///
    /// Asserted on the two labels being DIFFERENT and on the free-text one
    /// naming a technique, rather than on their exact wording: the wording is
    /// prompt text and may be tuned, but a build where a balloon line and a
    /// title card describe themselves identically is the bug coming back.
    #[test]
    fn a_free_standing_title_is_not_labelled_dialogue() {
        let dialogue = segment_kind(
            "dev.koharu.region.text",
            Some("dev.koharu.text.dialogue"),
        );
        let title = segment_kind(
            "dev.koharu.region.text",
            Some("dev.koharu.text.free-text"),
        );
        assert!(dialogue.is_some() && title.is_some());
        assert_ne!(
            dialogue, title,
            "a balloon line and a free-standing title must not describe themselves alike"
        );
        assert!(
            title.unwrap().contains("technique") || title.unwrap().contains("ability"),
            "the free-text label must tell the model a title card can be an ability name"
        );
        assert!(
            dialogue.unwrap().contains("balloon") || dialogue.unwrap().contains("spoken"),
            "and the dialogue label must say it is speech"
        );
    }

    /// A sound effect keeps its own word whatever role it carries -- an effect
    /// is an effect in a balloon or out of one.
    #[test]
    fn a_sound_effect_is_labelled_by_kind_alone() {
        for role in [
            None,
            Some("dev.koharu.text.dialogue"),
            Some("dev.koharu.text.free-text"),
        ] {
            assert_eq!(
                segment_kind("dev.koharu.region.onomatopoeia", role),
                Some("sound effect")
            );
        }
    }

    /// AN UNKNOWN ROLE IS OMITTED, NOT GUESSED. The wrong word is worse than no
    /// word -- that is what the caption label proved. A role this function does
    /// not recognise must disappear from the prompt rather than be approximated
    /// into the nearest label.
    #[test]
    fn an_unrecognised_role_yields_no_label() {
        assert_eq!(segment_kind("dev.koharu.region.text", None), None);
        assert_eq!(
            segment_kind("dev.koharu.region.text", Some("dev.koharu.text.whatever")),
            None
        );
        assert_eq!(segment_kind("dev.koharu.region.bubble", None), None);
    }

    /// A page with `sources` text layers on it, one per string. An empty string
    /// still gets a layer and a `SourceText`, because "blank after trimming" is
    /// a case the walk rejects for its own reason and the gate has to inherit.
    fn page_with(sources: &[&str]) -> (Snapshot, EntityId) {
        let mut session = Session::memory().unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        for source in sources {
            let content = edit.add_text_content(page, At::End).unwrap();
            edit.set(
                content,
                &koharu_scene::SourceText {
                    text: koharu_scene::Authored::user((*source).to_owned()),
                    language: None,
                },
            )
            .unwrap();
            edit.add_text_layer(
                page,
                At::End,
                content,
                &TextLayout {
                    origin: koharu_scene::Origin::User,
                    kind: TextLayoutKind::Paragraph,
                },
            )
            .unwrap();
        }
        session.commit(edit.finish().unwrap()).unwrap();
        (session.snapshot(), page)
    }

    fn input(scene: Snapshot, page: EntityId) -> StageInput {
        input_with_seed(scene, page, None)
    }

    fn input_with_seed(scene: Snapshot, page: EntityId, seed: Option<u32>) -> StageInput {
        StageInput::new(
            scene,
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
            Arc::from([] as [koharu_translator::TranslationContext; 0]),
            Arc::from([] as [koharu_translator::TranslationContext; 0]),
            seed,
            // No reader edits: translation tests never reach detection.
            Arc::from([] as [crate::CallerRegion; 0]),
            Arc::from([] as [crate::CallerRegion; 0]),
            // Not a joined page: an ordinary slice, so the size guard applies.
            false,
            Arc::from([] as [f64; 0]),
            None,
        )
    }

    fn processor() -> Processor {
        processor_with(TranslationConfig::default())
    }

    /// The retry's seed lands on the generation options the translator is
    /// HANDED -- the composed object, per `build_request`'s note one method up
    /// -- and an input without one changes nothing, which is every caller but
    /// the retry button. Proven able to fail by making `generation_for` drop
    /// the input read: the first assert went red at once.
    #[test]
    fn a_request_seed_lands_on_the_generation_and_absent_keeps_the_config() {
        let (scene, page) = page_with(&["猫"]);
        let processor = processor();
        assert_eq!(
            processor
                .generation_for(&input_with_seed(scene.clone(), page, Some(1234)))
                .seed,
            Some(1234),
            "the seed must reach the options the translator sees"
        );
        assert_eq!(
            processor.generation_for(&input(scene, page)).seed,
            None,
            "no caller seed leaves the config's own draw untouched"
        );
    }

    fn processor_with(config: TranslationConfig) -> Processor {
        let translator = koharu_translator::Translator::from_config(
            koharu_ml::Device::cpu(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        Processor::new(config, translator, false, false)
    }

    /// The COMPOSED request is what this asserts -- `build_request`, the exact
    /// object `process` hands the provider -- because a config lever that
    /// never reaches the request leaves every prompt-side unit test green
    /// while the shipping arm runs the baseline. Both
    /// arms, so the OFF default is pinned where the wiring lives and not only
    /// in the server's `cli.rs`.
    /// THE SEGMENT CONTEXT REACHES THE REQUEST, and the off arm carries none.
    ///
    /// `prompt.rs` owns whether an empty vec perturbs the prompt (it does not,
    /// asserted there). What only THIS crate can see is whether the flag moves
    /// the vec at all -- a lever that never arrives leaves every test in the
    /// translator green over a request that was never labelled.
    #[test]
    fn the_segment_context_flag_reaches_the_request() {
        let (scene, page) = page_with(&["\u{3042}", "\u{3044}"]);
        let input_off = input(scene, page);
        let targets_off = targets(&input_off, false, false).unwrap();
        let off = processor().build_request(&targets_off, &input_off);
        assert!(
            off.segment_context.is_empty(),
            "the default arm must send a bare segment list"
        );

        let (scene, page) = page_with(&["\u{3042}", "\u{3044}"]);
        let input_on = input(scene, page);
        let targets_on = targets(&input_on, false, false).unwrap();
        let on = processor_with(TranslationConfig {
            segment_context: true,
            ..TranslationConfig::default()
        })
        .build_request(&targets_on, &input_on);
        assert_eq!(
            on.segment_context.len(),
            on.segments.len(),
            "context is positional: it must be empty or exactly as long as the segments"
        );
    }

    #[test]
    fn the_containment_lever_reaches_the_built_request() {
        let (scene, page) = page_with(&["\u{3042}", "\u{3044}"]);
        let input_off = input(scene, page);
        let targets_off = targets(&input_off, false, false).unwrap();
        let off = processor().build_request(&targets_off, &input_off);
        assert!(
            !off.containment_clause,
            "the default arm must stay the baseline prompt"
        );

        let (scene, page) = page_with(&["\u{3042}", "\u{3044}"]);
        let input_on = input(scene, page);
        let targets_on = targets(&input_on, false, false).unwrap();
        let on = processor_with(TranslationConfig {
            containment_clause: true,
            ..TranslationConfig::default()
        })
        .build_request(&targets_on, &input_on);
        assert!(on.containment_clause, "the lever never reached the request");
        assert_eq!(
            on.segments, off.segments,
            "the lever must change the flag and nothing else about the request"
        );
    }

    #[test]
    fn a_page_with_no_text_at_all_has_nothing_to_translate() {
        // The measured case: 80 of 219 slices of a webtoon chapter detect no
        // region, so no text layer is ever minted and the page reaches this
        // stage bare.
        let (scene, page) = page_with(&[]);
        assert!(!processor().has_work(&input(scene, page)).unwrap());
    }

    #[test]
    fn a_page_whose_only_line_is_blank_has_nothing_to_translate() {
        let (scene, page) = page_with(&["   \n\t"]);
        assert!(!processor().has_work(&input(scene, page)).unwrap());
    }

    #[test]
    fn a_page_with_one_line_has_work() {
        let (scene, page) = page_with(&["\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}"]);
        assert!(processor().has_work(&input(scene, page)).unwrap());
    }

    #[test]
    fn a_blank_line_beside_a_real_one_still_has_work() {
        let (scene, page) = page_with(&["", "\u{3084}\u{3081}\u{308d}"]);
        assert!(processor().has_work(&input(scene, page)).unwrap());
    }

    /// The property the whole design rests on: `has_work` is not a second
    /// opinion about the page, it is the emptiness of the exact list `process`
    /// is about to translate.
    #[test]
    fn has_work_is_exactly_the_walk_process_runs() {
        let processor = processor();
        let cases: [&[&str]; 6] = [
            &[],
            &[""],
            &["  "],
            &["\u{3042}"],
            &["", "\u{3042}"],
            &["\u{3042}", "\u{3044}", ""],
        ];
        for sources in cases {
            let (scene, page) = page_with(sources);
            let input = input(scene, page);
            assert_eq!(
                processor.has_work(&input).unwrap(),
                !targets(&input, false, false).unwrap().is_empty(),
                "{sources:?}"
            );
        }
    }

    /// A page carrying the OCR stage's own lettering veto on chosen layers.
    ///
    /// `page_with` cannot express it, and the veto is one of the three arms of
    /// `unlettered_read` -- so without this helper the arm would ship with its
    /// `||` untested, which is the exact shape of an unwired fix.
    fn page_with_hidden(sources: &[&str], hidden: &[usize]) -> (Snapshot, EntityId) {
        let mut session = Session::memory().unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        for (index, source) in sources.iter().enumerate() {
            let content = edit.add_text_content(page, At::End).unwrap();
            edit.set(
                content,
                &koharu_scene::SourceText {
                    text: koharu_scene::Authored::user((*source).to_owned()),
                    language: None,
                },
            )
            .unwrap();
            let layer = edit
                .add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: koharu_scene::Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )
                .unwrap();
            if hidden.contains(&index) {
                edit.set(
                    layer,
                    &koharu_scene::Visibility {
                        origin: koharu_scene::Origin::User,
                        visible: false,
                        opacity: 1.0,
                    },
                )
                .unwrap();
            }
        }
        session.commit(edit.finish().unwrap()).unwrap();
        (session.snapshot(), page)
    }

    /// The source texts a page's targets carry, in order -- so an assertion can
    /// name WHICH read survived rather than only how many did. Counting alone
    /// would pass a filter that dropped the wrong one.
    fn sources_of(input: &StageInput, scoped: bool, skip: bool) -> Vec<String> {
        targets(input, scoped, skip)
            .unwrap()
            .into_iter()
            .map(|(_, source, _)| source)
            .collect()
    }

    /// ASSERTED ON `targets`, THE FUNCTION THE CALLER ACTUALLY CALLS.
    ///
    /// `illegible_text` has its own suite in `mod.rs` and every case there
    /// passed while this filter did not exist at all -- which is the point: a
    /// test of the halves leaves the `||` between them, and the wiring to the
    /// caller, entirely unexercised.
    ///
    /// `H` is a measured case, not an invented one: a residue admission on a
    /// Chinese test chapter read exactly that, was translated to `H`, and was
    /// then refused as "one or two Latin letters" after the model had already
    /// been paid for it.
    #[test]
    fn an_unlettered_read_is_kept_out_of_the_request() {
        let (scene, page) = page_with(&["\u{771F}\u{6C14}", "H"]);
        let input = input(scene, page);
        assert_eq!(
            sources_of(&input, false, false),
            vec!["\u{771F}\u{6C14}".to_owned(), "H".to_owned()],
            "the off arm must be byte-exact: both reads still go to the model"
        );
        assert_eq!(
            sources_of(&input, false, true),
            vec!["\u{771F}\u{6C14}".to_owned()],
            "with the filter on the illegible read is dropped and the real one kept"
        );
    }

    /// The third arm, and the one no string can reach.
    #[test]
    fn a_layer_the_ocr_stage_hid_is_kept_out_of_the_request() {
        let (scene, page) = page_with_hidden(&["\u{771F}\u{6C14}", "\u{4ED6}\u{7684}"], &[1]);
        let input = input(scene, page);
        assert_eq!(
            sources_of(&input, false, false).len(),
            2,
            "a hidden layer is still translated today"
        );
        assert_eq!(
            sources_of(&input, false, true),
            vec!["\u{771F}\u{6C14}".to_owned()],
            "the hidden layer's read is legible, so ONLY the visibility arm can drop it"
        );
    }

    /// The watermark arm, which needs `scoped` -- and which is the biggest of
    /// the three by a wide margin: 116 of the measured chapter's 331 regions, with
    /// an earlier, longer watermark list.
    #[test]
    fn a_site_watermark_is_kept_out_of_the_request() {
        // The site's plate, as the generic "latest free manga" watermark phrase.
        let (scene, page) = page_with(&[
            "\u{771F}\u{6C14}",
            "\u{6700}\u{65B0}\u{514D}\u{8D39}\u{6F2B}\u{753B}",
        ]);
        let input = input(scene, page);
        assert_eq!(
            sources_of(&input, true, false).len(),
            2,
            "today the plate is translated and only refused afterwards"
        );
        assert_eq!(
            sources_of(&input, true, true),
            vec!["\u{771F}\u{6C14}".to_owned()],
            "with the filter on the site's own plate never reaches the model"
        );
    }

    /// **THE SAFETY PROPERTY, and it is the reason the filter is shaped as a
    /// survivor test rather than a plain `retain`.**
    ///
    /// An emptied page makes `has_work` false, which SKIPS the stage, which
    /// means `Progress::Untranslated` never fires, which means
    /// `nothing_translated`'s 502 cannot fire -- and the renderer's
    /// `fallback_to_source_text` then paints the raw source in a translation
    /// font. The page would ship untranslated and look deliberate.
    ///
    /// So a page with nothing worth translating must keep every target it has
    /// today, in BOTH arms.
    #[test]
    fn the_filter_never_empties_a_page() {
        for sources in [&["H", "k"][..], &["\u{FF1F}"][..], &["H"][..]] {
            let (scene, page) = page_with(sources);
            let input = input(scene, page);
            assert_eq!(
                sources_of(&input, false, true),
                sources_of(&input, false, false),
                "every read is unlettered, so the filter must stand down: {sources:?}"
            );
            assert!(
                !targets(&input, false, true).unwrap().is_empty(),
                "an emptied page turns a 502 into a silent untranslated render: {sources:?}"
            );
        }
    }

    /// `has_work` is `!targets(..).is_empty()`, so the two cannot disagree by
    /// construction -- but that is exactly what makes an accidental change
    /// silent, and the filter is a change to the walk they share. Re-run with
    /// the filter ON, over the mixed cases where it actually removes something.
    #[test]
    fn has_work_is_unchanged_by_the_filter_on_every_page_shape() {
        let processor = processor();
        let cases: [&[&str]; 7] = [
            &[],
            &[""],
            &["H"],
            &["H", "k"],
            &["\u{771F}\u{6C14}"],
            &["\u{771F}\u{6C14}", "H"],
            &["", "H", "\u{771F}\u{6C14}"],
        ];
        for sources in cases {
            let (scene, page) = page_with(sources);
            let input = input(scene, page);
            let off = !targets(&input, false, false).unwrap().is_empty();
            let on = !targets(&input, false, true).unwrap().is_empty();
            assert_eq!(off, on, "the filter moved has_work on {sources:?}");
            assert_eq!(processor.has_work(&input).unwrap(), off, "{sources:?}");
        }
    }
}
