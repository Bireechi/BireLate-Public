use crate::Language;

/// Which style sentence the system prompt carries, between the task sentence and
/// the JSON contract.
///
/// A selector rather than a string, for exactly the reason
/// [`TranslationRequest::name_medium`] is a bool: every word the model reads
/// lives in `prompt.rs`, so a caller may pick an arm but may not compose one. A
/// caller that could pass its own sentence would be writing prompt text outside
/// the one funnel every provider shares.
///
/// Four arms because that one sentence carries three separable instructions --
/// what to preserve, whether sound effects belong on that list, and how long the
/// result may be -- and nothing but a measurement can say which of them earns
/// its tokens. It is the densest instruction in the prompt and the only one that
/// has never been varied.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StyleClause {
    /// The sentence that ships, unchanged.
    ///
    /// The default, so nothing renders differently until a caller asks for it:
    /// `an_unnamed_source_renders_the_prompt_it_always_has` pins the whole
    /// resulting prompt byte for byte.
    #[default]
    Shipped,
    /// The shipped sentence with sound effects struck off the preserve list.
    ///
    /// The hypothesis is that a PRESERVE list is the wrong home for them: in this
    /// field "preserve the sound effects" is idiom for leaving them in the source
    /// script, which is the *opposite* of this pipeline, which translates them.
    /// Not a bare deletion -- the Oxford "and" moves back onto
    /// "emphasis", so the sentence still reads as English.
    NoSfx,
    /// No style sentence at all: role, task, then the contract clauses.
    ///
    /// The control for the other two. Without it, a win for [`Self::NoSfx`] or
    /// [`Self::Sized`] cannot be told from the sentence simply being shorter.
    None,
    /// The shipped preserve list with only its concision tail replaced, by what a
    /// balloon actually does to a translation that outgrows it.
    ///
    /// Deliberately **not** a character count. A number invites trading meaning
    /// for the number, and it would bind unevenly across sources: dialogue p90 is
    /// 64 characters on manga-ja and 96 on webtoon-zh, so one cap is loose on one
    /// and a straitjacket on the other.
    Sized,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationRequest {
    pub segments: Vec<String>,
    pub source_language: Option<Language>,
    pub target_language: Language,
    pub instructions: Option<String>,
    pub context: Vec<TranslationContext>,
    /// Pinned per-series term renderings, shown to the model as law rather than
    /// precedent.
    ///
    /// Distinct from `context` on purpose: a context pair is *what an earlier
    /// page happened to say*, offered for continuity, while a glossary entry is
    /// *what this term is called*, to be used at every occurrence. The measured
    /// prototype rode the `instructions` free-text channel (an 8-term
    /// glossary), but instructions are snapshotted config -- changing them reloads
    /// the pipeline -- so the productised channel travels per request beside
    /// `context`, where a per-series store can vary it without a cold start.
    pub glossary: Vec<TranslationContext>,
    /// Per-segment context, positionally aligned with `segments`.
    ///
    /// **Empty, or exactly as long as `segments`.** A shorter vec would silently
    /// mislabel every segment after the gap, so `prompts` indexes it by position
    /// and treats a missing entry as empty rather than shifting.
    pub segment_context: Vec<SegmentContext>,
    /// Name the source medium -- manga / manhwa / manhua -- in the system
    /// prompt, always beside its language. Requires `source_language`; without
    /// one there is nothing to name and the prompt keeps its default noun.
    ///
    /// A `bool` rather than a string so the *wording* stays in `prompt.rs` with
    /// every other sentence the model reads. A caller that could pass its own
    /// noun would be composing prompt text outside the one funnel every
    /// provider shares, which is exactly what makes local and remote arms drift.
    pub name_medium: bool,
    /// Which style sentence the system prompt carries. See [`StyleClause`].
    ///
    /// Independent of `source_language` and `name_medium` by construction: those
    /// two vary the ROLE and TASK sentences, this one varies the sentence between
    /// them and the JSON contract. All twelve combinations render, and the style
    /// sentence is byte-identical across the three language arms -- which is the
    /// only thing that lets either measurement be attributed to the lever it
    /// names.
    pub style_clause: StyleClause,
    /// Whether the system prompt carries the containment sentence: each
    /// segment's translation covers that segment's own source and nothing of
    /// its neighbours', even where one sentence spans several segments.
    ///
    /// The measured defect it targets is
    /// RE-PARTITIONING, not missed adjacency: on every in-reach census exhibit
    /// the model had already joined the multi-segment sentence and then moved
    /// clauses between segments to satisfy English head-initial order --
    /// negation and matrix verbs climb into the earlier (Japanese-tail)
    /// segment, whose own clause is pushed down or dropped. Telling the model
    /// which segments continue is therefore the WRONG lever (the best
    /// detectable marking rule fires on ~85% of pages at sub-1% defect
    /// precision, and its false fires endorse the fabrication class); the
    /// surviving lever is this inverse, stated unconditionally when on.
    ///
    /// A `bool` and not a sentence, for `name_medium`'s reason: the wording
    /// lives in `prompt.rs` with every other sentence the model reads. `false`
    /// keeps the prompt byte-identical to one built before the field existed,
    /// so the baseline golden test does not move.
    pub containment_clause: bool,
}

impl TranslationRequest {
    #[must_use]
    pub fn new(
        segments: impl IntoIterator<Item = impl Into<String>>,
        target_language: Language,
    ) -> Self {
        Self {
            segments: segments.into_iter().map(Into::into).collect(),
            source_language: None,
            target_language,
            instructions: None,
            context: Vec::new(),
            glossary: Vec::new(),
            segment_context: Vec::new(),
            name_medium: false,
            style_clause: StyleClause::Shipped,
            containment_clause: false,
        }
    }

    /// The language the page is written in, named to the model instead of left
    /// as "the detected source language".
    ///
    /// Was `#[cfg(test)]`, exactly like `with_context` below, so every page ever
    /// translated through this crate has told the model to detect its own source
    /// -- the field, the serialisation into the user JSON and the system-prompt
    /// branch all existed and the only caller was a unit test.
    ///
    /// Ungated for a *caller that knows*, and the qualification is the whole
    /// point: `stages/ocr.rs` stamps a hardcoded `ja-JP` on every region and no
    /// OCR backend reports a language at all, so a pipeline that read it back out
    /// would assert "translate from Japanese" over a Chinese manhua. The honest
    /// producers are a per-request server field and a benchmark manifest that
    /// records what each source actually is.
    #[must_use]
    pub fn with_source_language(mut self, language: Language) -> Self {
        self.source_language = Some(language);
        self
    }

    /// Whether the system prompt names the medium as well as the language.
    ///
    /// Off by default, and only meaningful beside [`Self::with_source_language`]
    /// -- the noun is derived from the source language, so with none set there is
    /// nothing to derive it from.
    #[must_use]
    pub fn with_medium_term(mut self, name_medium: bool) -> Self {
        self.name_medium = name_medium;
        self
    }

    /// Which style sentence the prompt carries.
    ///
    /// [`StyleClause::Shipped`] by default, so this changes nothing until a
    /// caller names another arm. Unlike [`Self::with_medium_term`] it needs no
    /// companion setting -- the sentence it varies mentions neither the source
    /// language nor the medium, so every arm is meaningful on its own and beside
    /// any of the three language arms.
    #[must_use]
    pub fn with_style_clause(mut self, clause: StyleClause) -> Self {
        self.style_clause = clause;
        self
    }

    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Whether the prompt carries the containment sentence. See the field's doc.
    ///
    /// Composes with every other lever: it appends a conditional output rule
    /// after the fixed contract, exactly as `context` and `glossary` append
    /// theirs, so no arm's attribution moves. The single-segment repair paths
    /// clone the whole request (`template.clone()`), so a retry keeps the
    /// caller's arm without any code of its own -- which is the property that
    /// made this shape win over a per-segment annotation, a thing a
    /// one-segment retry structurally cannot carry.
    #[must_use]
    pub fn with_containment_clause(mut self, containment_clause: bool) -> Self {
        self.containment_clause = containment_clause;
        self
    }

    /// Prior source/target pairs, shown to the model as precedent.
    ///
    /// Was `#[cfg(test)]`, which is why `context` has been serialised into every
    /// prompt since it was written (`prompt.rs`) and has always arrived empty:
    /// the field existed, the plumbing existed, and the only two callers were
    /// unit tests. Ungated so a caller reading a serial can keep names,
    /// honorifics and register consistent across pages.
    #[must_use]
    pub fn with_context(mut self, context: impl IntoIterator<Item = TranslationContext>) -> Self {
        self.context = context.into_iter().collect();
        self
    }

    /// Pinned source/target term renderings for this series. See the field's
    /// doc for how this differs from [`Self::with_context`].
    #[must_use]
    pub fn with_glossary(mut self, glossary: impl IntoIterator<Item = TranslationContext>) -> Self {
        self.glossary = glossary.into_iter().collect();
        self
    }
}

/// What a provider actually returned for one [`TranslationRequest`].
///
/// `segments` is seeded from the request's own source text, so it is never
/// short and a segment the provider skipped reads exactly like one it
/// translated -- that bubble simply renders in the source language.
/// `untranslated` is the only record that it happened, which is why it travels
/// with the text rather than being guessed at later by comparing the two:
/// romaji, digits and untouched sound effects all translate to themselves.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Translated {
    pub segments: Vec<String>,
    /// Indices into `segments` that the provider never answered for, ascending.
    pub untranslated: Vec<usize>,
    /// Generation stopped on the token cap rather than on a stop token.
    ///
    /// **The local backend and the OpenAI-compatible family both report this.**
    /// That family is every provider routed through
    /// `ChatBackend` -- `openai-compatible` (which is how Ollama is reached),
    /// `openai`, `openrouter`, `deepseek` and `atlas-cloud` -- and it reads the
    /// `finish_reason` the response already carried and which was previously
    /// discarded while decoding.
    ///
    /// **`false` still means "not known" for the rest**: `lm-studio` answers with
    /// the Responses API's `output` array and carries no `finish_reason` at all,
    /// Claude uses `stop_reason` and Gemini `finishReason`, and none of those are
    /// decoded. An endpoint in the OpenAI family that simply omits the field is
    /// also "not known", and `truncated_by` warns once per process when that
    /// happens rather than letting the silence look like health.
    ///
    /// Why this matters more than a status field usually would: an overflowing
    /// page does *not* error. `prompt::translations` seeds
    /// its result from the source segments and fills in only the ids the model
    /// returned, so the surplus renders as untranslated Japanese that looks
    /// exactly like an OCR miss.
    pub truncated: bool,
    /// Reply entries naming an id that had already been answered, whose text was
    /// therefore dropped.
    ///
    /// **This is the discriminator `untranslated` cannot supply.** That list is
    /// derived from a "was this id filled in?" mask, so a segment nobody answered
    /// and a segment whose id was spent on somebody else's duplicate look
    /// identical by the time it is built -- and the two have opposite fixes. A
    /// non-zero count here says the model *answered* the page and mis-addressed
    /// the replies; a zero with a long `untranslated` says it stopped early.
    ///
    /// It is not a hypothetical failure. The local backend decodes under an
    /// llguidance grammar built from the per-page JSON schema, and that grammar
    /// constrains the *number* of entries but not their distinctness --
    /// llguidance does not implement `uniqueItems`. "Copy every input ID exactly
    /// once" is the one clause of the system prompt with nothing enforcing it.
    pub duplicate_ids: usize,
    /// Reply entries naming an id no segment has, whose text was likewise
    /// dropped.
    ///
    /// Counted apart from `duplicate_ids` because the two accuse the model of
    /// different things: a duplicate is a copying slip inside the id set it was
    /// given, an out-of-range id is an invented one. Under the local grammar the
    /// id is schema-constrained to the page's range and this should be
    /// structurally impossible, so any non-zero count on `--provider local` is
    /// evidence about the *grammar*, not about the model.
    pub out_of_range_ids: usize,
    /// Segments whose first reply arrived visibly cut mid-sentence -- the cut
    /// repair's own input population, counted BEFORE the repair flag is
    /// consulted so an off arm reports what it saw rather than nothing
    /// (the ambiguity this kills is the one `repair_cut_segments`'
    /// own telemetry comment records: "the retry never fired" and "the retry
    /// fired and failed" were indistinguishable from outside).
    pub cut_found: usize,
    /// Segments that SHIP visibly cut -- recounted from the final text after
    /// any repair, so a failed or rejected retry is included. Deliberately NOT
    /// the tracing line's `still_cut`, which counts only the retried-alone
    /// subset: the reader-facing question is "does this page carry a bubble
    /// that ends mid-word", and that is a property of what ships.
    pub still_cut: usize,
}

impl Translated {
    /// A reply from a provider that answers segment by segment, and so has no
    /// way to skip one.
    #[must_use]
    pub(crate) fn complete(segments: Vec<String>) -> Self {
        Self {
            segments,
            untranslated: Vec::new(),
            truncated: false,
            // Zero rather than "unknown": these providers are asked for one
            // segment at a time and answer positionally, so there is no id to
            // duplicate or to put out of range. Nothing is being assumed away.
            duplicate_ids: 0,
            out_of_range_ids: 0,
            // Filled in by `repair_cut_segments`, which runs after every
            // provider path; zero until then.
            cut_found: 0,
            still_cut: 0,
        }
    }
}

/// What the pipeline knows about ONE segment beyond its characters.
///
/// The model has always received a bare `{id, text}` list (`prompt.rs`), so a
/// name, a scream and a caption arrived indistinguishable, and a name the
/// artwork CANCELS arrived as an ordinary word. On one test page that is exactly
/// the failure: a character's true name is revealed beside a struck 冥霜之王, and
/// with no way to know that, the model returned an invented name.
///
/// Empty for every segment unless the pipeline fills it, and an empty one
/// serializes to nothing at all -- so a page carrying no context produces a
/// byte-identical prompt to one built before this existed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SegmentContext {
    /// The region kind this text was read from, as a short plain word --
    /// `dialogue`, `sound effect`, `display text`. Namespaced scene kinds are
    /// mapped to English by the pipeline: the model reads prose, not our
    /// identifiers.
    pub kind: Option<String>,
    /// Whether the ARTWORK strikes this text through -- a name being cancelled.
    /// Measured by `strike_ink` in the detection stage; before this field the
    /// fact reached the compositor and died there.
    pub struck: bool,
}

impl SegmentContext {
    /// Whether this carries nothing, and so must not perturb the prompt.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kind.is_none() && !self.struck
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TranslationContext {
    pub source: String,
    pub translation: String,
}

impl TranslationContext {
    #[must_use]
    pub fn new(source: impl Into<String>, translation: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            translation: translation.into(),
        }
    }
}
