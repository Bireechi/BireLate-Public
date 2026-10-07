use anyhow::Context;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::{Value, json};

use crate::{
    Language, SegmentContext, StyleClause, Translated, TranslationContext, TranslationRequest,
};

pub(crate) fn prompts(request: &TranslationRequest) -> anyhow::Result<(String, String)> {
    let input = TranslationInput {
        source_language: request.source_language,
        target_language: request.target_language,
        context: &request.context,
        glossary: &request.glossary,
        segments: request
            .segments
            .iter()
            .enumerate()
            .map(|(id, text)| {
                /* Indexed rather than zipped, so a `segment_context` that is
                 * empty (every page today) or short cannot shift labels onto
                 * the wrong segments -- it can only leave them unlabelled. */
                let context = request.segment_context.get(id);
                TranslationInputSegment {
                    id,
                    text,
                    kind: context.and_then(|value| value.kind.as_deref()),
                    struck: context.is_some_and(|value| value.struck),
                }
            })
            .collect(),
    };
    let user = serde_json::to_string(&input).context("failed to serialize translation input")?;
    Ok((translation_system_prompt(request), user))
}

/// Rebuilds the input order from the ids the model echoed back, and reports the
/// ids it never sent.
///
/// This is the one funnel every JSON-answering provider passes through, which
/// is why the misses are counted here. Nothing upstream can raise them: an id
/// left out keeps its source text, so the result is always exactly
/// `source_segments.len()` entries long and the caller's own segment-count
/// guard sees a complete reply.
///
/// The usual cause is a response cut off by the token cap, and that arrives
/// looking healthy: [`crate::json::from_str`] repairs truncated JSON rather
/// than failing on it, so a reply that stopped mid-array still parses into a
/// well-formed but short `translations` list. Omitted, duplicated and
/// out-of-range ids land here too, and are indistinguishable to a reader of the
/// page -- every one of them renders as untranslated source text.
///
/// They are no longer indistinguishable to a *caller*: `duplicate_ids` and
/// `out_of_range_ids` are counted separately from the miss list, because
/// `untranslated` is built from a filled-in mask and so reports the consequence
/// of a duplicate (some other id went unanswered) while saying nothing about the
/// cause. Those two causes want opposite fixes, so the count is the discriminator.
pub(crate) fn translations(
    provider: &str,
    text: &str,
    source_segments: &[String],
) -> anyhow::Result<Translated> {
    let output = crate::json::from_str::<TranslationOutput>(text)
        .with_context(|| format!("{provider} returned invalid translation JSON"))?;
    let mut segments = source_segments.to_vec();
    let mut translated = vec![false; source_segments.len()];

    /* Whether the reply arrived in input order, which nothing downstream can
     * ever ask again.
     *
     * The system prompt says "order does not matter" and this loop is where that
     * licence is exercised: ids are placed by index, so a shuffled reply and an
     * ordered one produce byte-identical output and the raw text is dropped on
     * the next line. The rate is therefore unobservable after this point, and it
     * decides a real question -- a positional reply schema (no ids at all) is
     * worth ~1.2s of a ~4.4s page, and is only safe if the model never reorders.
     * A shuffle rate above zero kills that idea; a run of zeroes over a volume
     * makes it worth building.
     *
     * Counted rather than assumed, and logged per page rather than aggregated,
     * because the interesting question is what fraction of PAGES reorder. Only
     * ids that were actually placed count: a duplicate or an out-of-range id is
     * already reported by the miss list and would otherwise be double-counted
     * here as disorder it did not cause. */
    let mut previous: Option<usize> = None;
    let mut out_of_order = 0usize;
    let mut placed = 0usize;

    /* The two ways a reply can be wrong rather than short, counted apart from
     * each other and from the miss list.
     *
     * `untranslated` is derived from the `translated` mask below, and a mask
     * cannot answer this: the id a duplicate landed on is marked done, so the
     * duplicate leaves no trace at all and the id nobody answered shows up as an
     * ordinary miss with no cause attached. "The model never answered these 7"
     * and "the model answered 7 of them twice" are then the same report, and
     * they have opposite fixes -- raise the token budget, or repair the id
     * discipline in the prompt.
     *
     * It is the prompt clause with nothing behind it. The local backend decodes
     * under an llguidance grammar built from the page's JSON schema, and that
     * grammar fixes the NUMBER of entries but not their distinctness --
     * llguidance implements no `uniqueItems`. "Copy every input ID exactly once"
     * is therefore advice, and this is the counter that says whether it was
     * taken. An out-of-range id is separately schema-constrained, so a non-zero
     * count there under `local` is evidence about the grammar rather than the
     * model, which is why the two are not summed into one number. */
    let mut duplicate_ids = 0usize;
    let mut out_of_range_ids = 0usize;

    for translation in output.translations {
        // Split out of the old single condition rather than counted beside it:
        // the `else` arms must not touch `previous` or `placed`, or a duplicate
        // would register as disorder it did not cause. See the comment above.
        if translation.id >= segments.len() {
            out_of_range_ids += 1;
        } else if translated[translation.id] {
            duplicate_ids += 1;
        } else {
            if previous.is_some_and(|last| translation.id < last) {
                out_of_order += 1;
            }
            previous = Some(translation.id);
            placed += 1;
            segments[translation.id] = without_nul(provider, translation.id, translation.text);
            translated[translation.id] = true;
        }
    }

    if out_of_order > 0 {
        tracing::info!(
            provider,
            out_of_order,
            placed,
            "the reply did not arrive in input order"
        );
    } else {
        // The negative is the result here, so it has to be logged too: a page
        // that is silent about ordering is indistinguishable from one where this
        // never ran, which is the shape of evidence this project keeps rejecting.
        tracing::debug!(provider, placed, "the reply arrived in input order");
    }

    if duplicate_ids > 0 || out_of_range_ids > 0 {
        tracing::warn!(
            provider,
            duplicate_ids,
            out_of_range_ids,
            segments = segments.len(),
            "the reply mis-addressed some entries and their text was dropped"
        );
    } else {
        // Logged like the ordering negative above, and for the same reason: this
        // is the only place the grammar's one unenforced clause is checked, and a
        // page silent about it is indistinguishable from a build where the check
        // is not present. A run of zeroes over a volume is the result.
        tracing::debug!(
            provider,
            segments = segments.len(),
            "every reply id was in range and answered once"
        );
    }

    let untranslated = translated
        .iter()
        .enumerate()
        .filter_map(|(id, done)| (!done).then_some(id))
        .collect::<Vec<_>>();
    if !untranslated.is_empty() {
        tracing::warn!(
            provider,
            untranslated = untranslated.len(),
            segments = segments.len(),
            "segments left holding their source text and will render untranslated"
        );
    }

    Ok(Translated {
        segments,
        untranslated,
        truncated: false,
        duplicate_ids,
        out_of_range_ids,
        // The cut verdict is `repair_cut_segments`' to write, downstream of
        // every reply parse; zero here means only "not yet examined".
        cut_found: 0,
        still_cut: 0,
    })
}

/// Removes NUL from one translated segment, because a single one kills the page.
///
/// `koharu-scene`'s `validate_authored_text` refuses any text containing `\0`,
/// so a NUL anywhere in a reply fails the scene commit, fails the translation
/// stage, and throws away the whole page -- detection, OCR and inpainting
/// included. The reader gets an error instead of a partly translated page. It is
/// not rare enough to ignore: once in 41 consecutive real pages.
///
/// **It is an ESCAPE, not a decoding fault, and the difference matters.** The
/// output schema types a translation as a bare JSON `string` with no `pattern`,
/// so the constrained-decoding grammar carries that escape like any other and
/// the model is free to emit it. Both parse paths then hand it back as a real
/// NUL: strict `serde_json` decodes the escape without complaint, and
/// `json.rs`'s repairing parser reaches `char::from_u32(0)`, which is
/// a `Some` holding a real NUL rather than the replacement character.
///
/// A *raw* NUL byte cannot come from the local model, and it is worth recording
/// why so that nobody re-opens token decoding: `token_to_piece_bytes` recovers
/// its buffer with `CString::from_raw`, which stops at the first NUL, so a
/// byte-fallback `<0x00>` token yields an EMPTY piece and the following
/// `truncate` is a no-op -- the byte is dropped long before the parser.
/// `koharu-llama`'s `decode_piece` is not implicated either: it fixes tokens
/// being *dropped*, and
/// `encoding_rs` emits U+FFFD for bad bytes, never NUL. The raw route is still
/// handled here because a remote provider can put a raw NUL in its response
/// body, where the repairing parser takes it verbatim.
///
/// This deliberately mirrors the validator's rule exactly rather than stripping
/// control characters generally. Only NUL can fail the commit, so only NUL is a
/// correctness fix; widening it would be inventing a policy on no evidence. The
/// other half of that rule, the 16 MiB length cap, is unreachable while
/// `max_tokens` is capped per page.
///
/// Stripped rather than replaced with U+FFFD: a NUL is not a character the model
/// meant, so dropping it loses nothing, while a replacement char would letter a
/// visible tofu box onto the artwork.
fn without_nul(provider: &str, id: usize, text: String) -> String {
    if !text.contains('\0') {
        return text;
    }
    let cleaned = text.replace('\0', "");
    tracing::warn!(
        provider,
        id,
        removed = text.len() - cleaned.len(),
        text = %text.escape_debug(),
        "stripped NUL from a translation; it would have failed the scene commit"
    );
    cleaned
}

/// The escape letters a translation may use after a backslash.
///
/// **The only control character a manga caption needs is a line break.** The JSON
/// default is `nrbtf\"u`, and every letter in it that also begins a LaTeX command is
/// a trap: a model writing `\rightarrow` inside a JSON string emits a *legal* `\r`,
/// so a spec-correct parser reads U+000D and eats the `r` with it. That is not a
/// parser bug and no parser can fix it -- 212 of the 213 non-newline control
/// characters in the test corpus are a destroyed command.
///
/// Dropping `r`, `b`, `t` and `f` makes those four structurally unrepresentable:
/// llguidance masks the token and the model's only remaining way to write a
/// backslash is `\\`, which round-trips. It is the same rescue that already happens
/// by accident for `\l` in `\leftarrow`, made deliberate.
///
/// **What each dropped letter costs, measured rather than assumed.** `\r` and `\b`:
/// nothing -- every occurrence in the test corpus is a destroyed `\rightarrow` or
/// `\blacksquare`. `\f`: nothing, zero occurrences. `\t`: one, a tab the model put
/// where a space belonged, which is junk rather than something worth preserving
/// -- and dropping it also closes `\times`, `\text` and `\theta`.
///
/// `n` stays: **2,265** translated fields carry a real line break. `\\`, `\"` and
/// `\uXXXX` stay because dialogue contains backslashes, quotes and arbitrary
/// Unicode. Note that a u-escape can still spell a carriage return, so this bounds
/// ACCIDENT, not the alphabet.
pub(crate) const LOCAL_ALLOWED_ESCAPES: &str = "n\\\"u";

/// [`output_schema`] plus the llguidance override, for the **local** backend only.
///
/// **Deliberately not folded into `output_schema`.** That value is also sent to
/// Gemini as `response_json_schema` and to OpenAI-compatible providers inside
/// `response_format`, and `x-guidance` is not a JSON Schema keyword -- a provider
/// that validates its input strictly would reject the request outright. Only
/// llguidance reads it, so only the path that uses llguidance carries it.
///
/// llguidance strips the key before compiling and deserializes it into
/// `JsonCompileOptions`, which is `#[serde(default, deny_unknown_fields)]` -- so
/// every other option keeps the default koharu already relied on, and a typo here
/// fails loudly rather than being ignored.
pub(crate) fn local_output_schema(expected: usize) -> Value {
    let mut schema = output_schema(expected);
    schema["x-guidance"] = json!({ "json_allowed_escapes": LOCAL_ALLOWED_ESCAPES });
    schema
}

pub(crate) fn output_schema(expected: usize) -> Value {
    json!({
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "minItems": expected,
                "maxItems": expected,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": expected.saturating_sub(1),
                            "description": "The ID copied from the corresponding input segment."
                        },
                        "text": {
                            "type": "string",
                            // A reply of "" satisfied every other clause of this
                            // schema, and that is a page defect rather than a
                            // formatting one. `minItems`/`maxItems` force the
                            // model to emit exactly one entry per segment, so a
                            // segment it does not want to answer cannot be
                            // omitted -- but it CAN be answered with nothing, and
                            // the entry count still checks out.
                            //
                            // Downstream that is invisible and expensive. The
                            // erase mask is written during detection, long before
                            // any translation exists, so an empty answer leaves a
                            // region that was rubbed off the page and lettered
                            // with nothing: a blank patch, 502 px at the smallest
                            // measured and 92,296 px at the largest. Measured on
                            // 2,337 regions of one volume with the story window
                            // on, 7 regions came back empty and every completeness
                            // counter -- `untranslated`, `truncated`,
                            // `duplicate_ids`, `out_of_range_ids` -- read clean on
                            // all 7. One of them deleted a page's punchline.
                            //
                            // Vetoing the erase instead is not available: the
                            // scheduler runs inpainting BEFORE translation on the
                            // large majority of pages (3,322 of 3,892 recorded
                            // four-stage pages), so by the time the empty answer
                            // exists the pixels are already gone.
                            //
                            // NB this constrains the *decoder* only on the local
                            // backend, which decodes under an llguidance grammar
                            // compiled from this schema. Remote providers get it
                            // as advice, exactly like the id-distinctness clause
                            // that `duplicate_ids` exists to count.
                            "minLength": 1,
                            "description": "The translation of the input segment with this ID. Never empty."
                        }
                    },
                    "required": ["id", "text"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["translations"],
        "additionalProperties": false
    })
}

/// The noun the prompt uses when nothing better is known, and what it has always
/// used: every page this crate has ever translated was introduced as manga.
const DEFAULT_MEDIUM: &str = "manga";

/// What the *system prompt* calls a source language, which is deliberately not
/// always what `Display` calls it.
///
/// `Language::ChineseSimplified` displays as "Simplified Chinese" and is what a
/// bare `zh` parses to -- and `zh` is the only Chinese tag anything upstream
/// produces: the extension latches one, and the benchmark manifest records one
/// per source. Rendering that into the prompt tells the model a traditional page
/// is simplified, which is a claim about pixels nobody checked. "Chinese" is
/// true of both and loses nothing the model needs; the BCP-47 tag in the user
/// JSON still carries the precise variant for anything that wants it.
///
/// Only the SOURCE is coarsened. A target of `zh-CN` is a caller asking for
/// simplified output on purpose, so flattening that would discard a real
/// instruction rather than an unfounded one.
fn source_language_name(language: Language) -> String {
    match language {
        Language::ChineseSimplified | Language::ChineseTraditional => "Chinese".to_owned(),
        other => other.to_string(),
    }
}

/// The medium noun for a source language, **always paired with the language it
/// belongs to** where the pairing does any work.
///
/// "manhwa" and "manhua" differ by one letter, name the comics of two different
/// countries, and are widely swapped in the wild -- so a model that has them the
/// wrong way round still reads the right country off the adjacent word. "manga"
/// has no such twin, and is already the noun in the unqualified prompt, so
/// Japanese pages get today's sentence unchanged; that arm is a no-op on
/// Japanese by construction and the point of it lies on the other two.
///
/// Anything else keeps [`DEFAULT_MEDIUM`]: inventing a noun for a language with
/// no comics tradition of its own would be asserting more than is known, which
/// is the same fault as naming the wrong Chinese.
fn medium_term(language: Option<Language>) -> &'static str {
    match language {
        Some(Language::Japanese) => "manga",
        Some(Language::Korean) => "Korean manhwa",
        Some(Language::ChineseSimplified | Language::ChineseTraditional) => "Chinese manhua",
        _ => DEFAULT_MEDIUM,
    }
}

/// The style sentence for one [`StyleClause`], **including the trailing space
/// that separates it from the JSON contract**.
///
/// The space belongs to the sentence rather than to the format string because
/// [`StyleClause::None`] deletes the sentence, and a separator left behind by a
/// deleted sentence is a doubled space in the middle of the prompt. Rendering
/// the empty arm as `""` makes the deletion exact instead of nearly exact.
///
/// Every word the model can read lives here, in the one funnel every provider
/// shares. `replay.rs` picks an arm by name and never composes one, so a local
/// and a remote run of the same arm cannot drift.
///
/// The differences, stated once so a reader does not have to diff four literals:
///
/// - [`StyleClause::NoSfx`] strikes sound effects off the preserve list and moves
///   the Oxford "and" onto "emphasis". The concision tail is untouched, so it is
///   the shipped sentence minus exactly one item.
/// - [`StyleClause::Sized`] keeps the preserve list verbatim and replaces only
///   the concision tail, with what a balloon actually does to a long line rather
///   than with a character count.
/// - [`StyleClause::None`] removes both.
///
/// Between them the two halves of the sentence are varied independently, which
/// is what makes a win attributable to one of them.
fn style_clause_text(clause: StyleClause) -> &'static str {
    match clause {
        StyleClause::Shipped => concat!(
            "Preserve character voice, emotional tone, relationship nuance, emphasis, and sound ",
            "effects while keeping wording concise enough for speech bubbles. "
        ),
        StyleClause::NoSfx => concat!(
            "Preserve character voice, emotional tone, relationship nuance, and emphasis while ",
            "keeping wording concise enough for speech bubbles. "
        ),
        StyleClause::None => "",
        StyleClause::Sized => concat!(
            "Preserve character voice, emotional tone, relationship nuance, emphasis, and sound ",
            "effects while keeping each translation as short as the meaning allows: a speech ",
            "balloon has room for roughly one short line, and a longer translation is lettered ",
            "smaller. "
        ),
    }
}

/// The three arms this can render, and the only two levers that pick between
/// them, are `source_language` and `name_medium`:
///
/// - neither: "from the detected source language", "professional manga
///   translator". Byte for byte what this has always emitted, and
///   `an_unnamed_source_renders_the_prompt_it_always_has` pins the whole string
///   so it cannot drift by a character.
/// - `source_language`: the same sentence with the language named.
/// - both: as above, plus the medium noun in place of the default one.
///
/// Everything from "Preserve character voice" onward is medium-neutral and is
/// identical in all three.
///
/// `style_clause` is a **third, orthogonal** lever over that neutral part: it
/// varies the one sentence between the task and the JSON contract, mentions
/// neither the language nor the medium, and defaults to the shipped wording. So
/// all twelve combinations render, the style sentence is byte-identical across
/// the three language arms, and the language sentences are byte-identical across
/// the four style arms -- which is what keeps two separate measurements from
/// contaminating each other.
fn translation_system_prompt(request: &TranslationRequest) -> String {
    let source = request
        .source_language
        .map(source_language_name)
        .unwrap_or_else(|| "the detected source language".to_owned());
    let medium = if request.name_medium {
        medium_term(request.source_language)
    } else {
        DEFAULT_MEDIUM
    };
    let mut prompt = format!(
        concat!(
            "You are a professional {medium} translator. ",
            "Translate every input segment from {source} into natural {target}. ",
            "{style}",
            "Each input segment has a numeric `id`. Return only a JSON object whose ",
            "`translations` array contains one object with `id` and translated `text` for every ",
            "input segment. Copy every input ID exactly once; order does not matter. Never merge, ",
            "split, omit, or add segments.",
            // The model was INVENTING LaTeX for glyphs the page already had: an
            // arrow between two words came back as `$\rightarrow$` and a `□` as
            // `$\blacksquare$`. Nothing in those sources has a backslash --
            // it is reaching for markup it knows from training, for a character
            // that renders perfectly well on its own.
            //
            // AFTER the segment rules and not beside `{style}`, deliberately. The
            // four style arms are extracted by `style_clause_of` as "everything
            // between the target-language sentence and `Each input segment`", so a
            // clause placed there is read as part of an experimental lever: it
            // silently became a fifth style arm and broke three tests that exist
            // to keep the arms attributable. This is a correctness rule, not an
            // arm, and it belongs with the other output rules.
            //
            // It carries NO backslash of its own. The system prompt is the only
            // place the model meets one before inventing one, and an example of the
            // wrong form is an exemplar of the wrong form.
            //
            // It names the `text` field rather than saying "plain text", because
            // the reply IS JSON and "write plain text" contradicts the sentence
            // before it.
            " Translated `text` must contain no LaTeX, Markdown, or other markup: ",
            "write every symbol as the literal character it is.",
            // This sentence is a repair for a GRAMMAR collision,
            // not a style preference. A straight ASCII quote is the one byte
            // that ends a JSON string, so when the English wants a citation
            // mid-sentence -- `That "request"...` -- the natural straight-quote
            // token TERMINATES the `text` value instead, and the pinned-length
            // reply grammar then forces a fresh id: one source's translation
            // lands across two slots and every later slot inherits its
            // predecessor's sentence, invisible to every counter. Measured:
            // cut sites are 20-65x enriched for
            // quote-bracket sources, and on the first 8 independent seed draws
            // 63.6% of prefix-agreeing completions place a quote or emphasis
            // mark at the exact cut offset. Typographic quotes are ordinary
            // string bytes -- no escape, no terminator -- and the model already
            // reaches for them spontaneously when recovering from a cut.
            //
            // Like the markup rule above: placed with the output rules, never
            // beside `{style}` (it would become a fifth style arm), and it
            // carries no ASCII straight quote of its own -- an example of the
            // wrong form is an exemplar of the wrong form.
            " When the translation needs quotation marks or apostrophes, write ",
            "the typographic characters \u{201c} \u{201d} \u{2018} \u{2019} and never the ASCII ",
            "straight forms."
        ),
        medium = medium,
        source = source,
        style = style_clause_text(request.style_clause),
        target = request.target_language,
    );

    /* The sentence is the INVERSE of the obvious direction (marking which
     * segments continue). The measured slide class is not missed
     * adjacency -- on every in-reach exhibit the model had already joined the
     * multi-segment sentence -- it is re-partitioning under English
     * head-initial order: the verb, negation or matrix clause climbs into the
     * earlier (Japanese-tail) segment and that segment's own clause is pushed
     * down or dropped. So the prompt does not mark continuations (sub-1%
     * defect precision for the best detectable rule, and a false mark endorses
     * the stammer-fabrication class); it forbids the redistribution instead.
     *
     * Conditional on the request flag, so the baseline prompt stays
     * byte-identical to one built before the field existed and the golden
     * whole-string test below does not move. Placed with the conditional
     * sentences and never beside `{style}`: `style_clause_of` reads everything
     * between the target-language sentence and `Each input segment` as the
     * style arm, and a clause placed there silently becomes a fifth arm --
     * the lesson recorded above.
     *
     * Like every sentence here it carries no ASCII straight quote and no
     * backslash of its own. */
    if request.containment_clause {
        prompt.push_str(
            " When consecutive input segments form one continuing sentence, translate each segment so its `text` covers only that segment's own words: do not move a clause into a neighbouring segment to improve the English flow, and prefer a slightly awkward split over redistributed content.",
        );
    }

    if !request.context.is_empty() {
        prompt.push_str(
            " Use the supplied context only to preserve terminology, character voice, and dialogue continuity. Do not translate or return the context entries.",
        );
    }

    /* The wording mirrors the measured free-text instructions prototype
     * ("always render these terms exactly as given, in every grammatical
     * position"), which delivered the pinned rendering at every registered
     * site over two test chapters. Kept conditional so a page
     * without a glossary carries a byte-identical prompt to one built before
     * the field existed. */
    if !request.glossary.is_empty() {
        prompt.push_str(
            " A terminology glossary for this series is supplied in the input. Always render each glossary source term exactly as its given translation, in every grammatical position, inflected only as the sentence requires. Do not translate or return the glossary entries themselves.",
        );
    }

    /* Conditional, so a page carrying no segment context builds a byte-identical
     * prompt -- the same discipline as the glossary and containment sentences
     * above, and what keeps the golden whole-string test still pinned.
     *
     * Two sentences, because they answer two different questions. `kind` says
     * what a line IS. `struck` says the ARTWORK cancels it -- and the
     * instruction attached to it is the whole point: a struck name has a
     * replacement somewhere else on the page, and the replacement is the thing
     * the model was inventing. It is told to render meaning rather than coin a
     * name, because an invented name is neither the meaning nor the sound.
     *
     * ## The same-character clause: removed, then RESTORED
     *
     * *"Translate a cancelled name and its replacement as the names of the same
     * character"* was removed and put back, and the round trip is the record
     * worth keeping.
     *
     * **Why it was removed:** it is a bleed. Told the two segments name one
     * being, the model harmonises them: a replacement name gained a "King"
     * imported from the 王 of the struck 冥霜之王, although its own source has
     * no 王 in it. Removing it also fixed two bleeds nobody had attributed to
     * it, one of them a spurious "Five" that had leaked from 五灯齐明 on a
     * DIFFERENT page.
     *
     * **Why it came back:** without it the model falls back on the permission
     * this sentence then still granted -- *"a transcription of its sound"* --
     * and on a short opaque compound it takes it, producing pinyin syllables
     * that are legitimate under the rule and meaningless to an English reader.
     * Compared on the rendered pages, a name a reader can parse beats a correct
     * transcription they cannot.
     *
     * **So the cost is known and accepted, not overlooked.** The clause buys a
     * readable name on the device and pays for it in cross-segment harmonising
     * elsewhere. Do not remove it again as an obvious cleanup -- that experiment
     * has been run, in both directions, and the pages were looked at.
     *
     * ## No transliterated ability names
     *
     * This sentence used to end *"never invent a name that is neither a
     * translation of the source nor a transcription of its sound"*, and that
     * trailing permission is what produced the defect: told a transcription is
     * acceptable, the model reaches for one on any compound it cannot gloss
     * confidently. Measured through the SEAM (the shipping path) it turned a
     * four-character technique name into pinyin syllables where the arm
     * WITHOUT this sentence rendered a meaning-based English name close to the
     * licensed English edition's. Another name went the same way.
     *
     * So the permission is withdrawn and transliteration is named and refused.
     * Do not reintroduce "or a transcription of its sound" as a softening: it
     * reads as a reasonable fallback and it is the whole mechanism.
     *
     * The rest of the sentence works in every arm measured and is not in
     * question: the same technique name improved from a half-transliterated
     * rendering to a meaning-based one with the same-character clause in and
     * with it out, on the per-page arm.
     *
     * No ASCII straight quote and no backslash, like every sentence here. */
    if request.segment_context.iter().any(|value| !value.is_empty()) {
        prompt.push_str(
            " Some input segments carry extra fields describing what they are. A `kind` field names the sort of text -- translate a sound effect as a sound effect and dialogue as speech. A `struck` field set to true means the artwork draws a line THROUGH that text: the story is cancelling that name, and another segment on the same page is the replacement it is being cancelled in favour of. Translate a cancelled name and its replacement as the names of the same character. Render the name of a person, a place, a technique or an ability by what it MEANS in the target language, or by its established English form: do not spell out its source pronunciation in target-language syllables, and do not invent a name unrelated to the source.",
        );
    }

    if let Some(instructions) = request
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|instructions| !instructions.is_empty())
    {
        prompt.push_str(" Additional instructions: ");
        prompt.push_str(instructions);
    }
    prompt
}

#[derive(Serialize)]
struct TranslationInput<'a> {
    source_language: Option<Language>,
    target_language: Language,
    context: &'a [TranslationContext],
    /// Skipped when empty, unlike `context`: `context` has serialized an empty
    /// array into every prompt since it was written, so its shape is pinned by
    /// measurement, while a glossary-less page should read byte-identically to
    /// one built before this field existed.
    #[serde(skip_serializing_if = "<[TranslationContext]>::is_empty")]
    glossary: &'a [TranslationContext],
    segments: Vec<TranslationInputSegment<'a>>,
}

#[derive(Serialize)]
struct TranslationInputSegment<'a> {
    id: usize,
    text: &'a str,
    /// What kind of text this is, in plain words. Omitted entirely when unknown,
    /// which is every page built before segment context existed and every page
    /// with the flag off -- the serialization is then byte-identical to the
    /// two-field form.
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<&'a str>,
    /// Whether the artwork strikes this text through. Omitted when false, for
    /// the reason above; `false` is overwhelmingly the common case and a
    /// `"struck": false` on every segment would be noise the model must read.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    struck: bool,
}

/// A reply's segments, **dropping any that do not deserialize rather than
/// failing the page**.
///
/// The list is repaired before it gets here -- [`crate::json::from_str`] closes
/// a truncated reply instead of refusing it -- and the whole design downstream
/// depends on a short list being survivable: `translations` below seeds its
/// result from the source segments, so ids the reply never reached keep their
/// Japanese, land in `untranslated`, and are reported. That is the intended
/// degradation for an overflowing page, and the token budget's `truncated`
/// report exists to make it visible.
///
/// **One shape of truncation escaped it and killed the page instead.** The
/// repairing parser fills a key whose value the input never reached with
/// `Value::Null` and returns, so a cut landing inside a segment's *scaffolding*
/// -- after `{`, after `"id"`, after either colon, in the id digits, after the
/// comma, or after `"text"` -- yields `{"id":5,"text":null}` or `{"id":5}`.
/// Either fails `TranslationOutputSegment`, and because the failure was on the
/// `Vec` it took the entire reply with it: every bubble on the page rendered
/// untranslated, after detection, OCR, inpainting and generation had all been
/// paid for. A cut two characters later, inside the text string, degraded
/// gracefully. The difference was where the model happened to stop.
///
/// Deserializing element-wise closes that. A complete reply is unaffected --
/// every element converts, and the result is identical to the derived
/// implementation. Only an element that cannot be read at all is dropped, and
/// dropping it is precisely what makes it an ordinary miss.
///
/// Not `Vec<Option<..>>` with a filter: `Option` would swallow a *well-formed*
/// element carrying a null field just as quietly, and this must only ever
/// forgive a genuinely unreadable one.
#[derive(Debug)]
struct TranslationOutput {
    translations: Vec<TranslationOutputSegment>,
}

impl<'de> Deserialize<'de> for TranslationOutput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            translations: Vec<Value>,
        }

        let raw = Raw::deserialize(deserializer)?;
        let total = raw.translations.len();
        let translations = raw
            .translations
            .into_iter()
            .filter_map(|value| serde_json::from_value::<TranslationOutputSegment>(value).ok())
            .collect::<Vec<_>>();
        if translations.len() != total {
            // Warn rather than swallow: the ids are still reported as misses by
            // `translations`, but this is the only line that says the reply was
            // malformed rather than merely short.
            tracing::warn!(
                dropped = total - translations.len(),
                segments = total,
                "discarded unreadable translation segments; they will render as source text"
            );
        }
        Ok(Self { translations })
    }
}

#[derive(Debug, Deserialize)]
struct TranslationOutputSegment {
    #[serde(deserialize_with = "deserialize_segment_id")]
    id: usize,
    text: String,
}

fn deserialize_segment_id<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SegmentId {
        Number(usize),
        String(String),
    }

    match SegmentId::deserialize(deserializer)? {
        SegmentId::Number(id) => Ok(id),
        SegmentId::String(id) => id.trim().parse().map_err(de::Error::custom),
    }
}

#[cfg(test)]
mod tests {

    /// THE OFF ARM IS BYTE-EXACT, and on this crate that is the whole safety
    /// property of the change: a prompt edit re-rolls every page, so a request
    /// carrying no segment context must serialize to the bytes it always did.
    #[test]
    fn a_request_without_segment_context_serializes_as_it_always_did() {
        let request = TranslationRequest::new(["\u{3042}", "\u{3044}"], Language::English);
        let (system, user) = prompts(&request).expect("prompts must build");
        assert!(
            !user.contains("kind") && !user.contains("struck"),
            "an empty segment context must add no field to the input: {user}"
        );
        assert!(
            !system.contains("struck"),
            "and must append no sentence to the system prompt"
        );
    }

    /// THE FACTS REACH THE MODEL. Both of them, and the struck one carries the
    /// instruction that is the point of segment context: a cancelled name has a
    /// replacement elsewhere on the page, and the model must not coin one.
    #[test]
    fn segment_context_reaches_the_prompt() {
        let mut request =
            TranslationRequest::new(["\u{51a5}\u{971c}\u{4e4b}\u{738b}", "\u{88c2}\u{98ce}"], Language::English);
        request.segment_context = vec![
            SegmentContext {
                kind: Some("dialogue or caption".to_owned()),
                struck: true,
            },
            SegmentContext {
                kind: Some("dialogue or caption".to_owned()),
                struck: false,
            },
        ];
        let (system, user) = prompts(&request).expect("prompts must build");
        assert!(user.contains("\"struck\":true"), "the strike must reach the input: {user}");
        assert!(
            user.matches("\"struck\"").count() == 1,
            "and only on the struck segment -- false is omitted, not serialized: {user}"
        );
        assert!(
            user.contains("\"kind\":\"dialogue or caption\""),
            "the kind must reach the input: {user}"
        );
        assert!(
            system.contains("do not spell out its source pronunciation"),
            "the no-transliteration rule must be in the prompt"
        );
        assert!(
            !system.contains("transcription of its sound"),
            "and the permission it replaced must NOT come back: it reads as a              reasonable fallback and it is the whole mechanism"
        );
    }

    /// A SHORT CONTEXT VEC MUST NOT SHIFT LABELS. The context is positional, so
    /// the failure to guard against is not "missing" but "applied to the wrong
    /// segment" -- which would tell the model a different line is cancelled.
    #[test]
    fn a_short_segment_context_leaves_later_segments_unlabelled() {
        let mut request = TranslationRequest::new(["a", "b", "c"], Language::English);
        request.segment_context = vec![SegmentContext {
            kind: None,
            struck: true,
        }];
        let (_, user) = prompts(&request).expect("prompts must build");
        assert_eq!(
            user.matches("\"struck\"").count(),
            1,
            "exactly the first segment is struck, and no later one inherits it: {user}"
        );
    }
    use super::*;

    #[test]
    fn parses_plain_json_and_markdown_fences() {
        let source = ["one".to_owned(), "two".to_owned()];
        let expected = vec!["hello".to_owned(), "world".to_owned()];
        for response in [
            r#"{"translations":[{"id":0,"text":"hello"},{"id":1,"text":"world"}]}"#,
            "```json\n{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"}]}\n```",
            "```JSON\n{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"}]}\n```",
            "```\n{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"}]}\n```",
        ] {
            assert_eq!(
                translations("test", response, &source).unwrap().segments,
                expected
            );
        }
    }

    #[test]
    fn repairs_malformed_llm_json() {
        let source = ["one".to_owned(), "two".to_owned()];
        let expected = vec!["hello".to_owned(), "world".to_owned()];
        for response in [
            r#"{translations: [{id: 0, text: 'hello'}, {id: 1, text: 'world'},],}"#,
            r#"Here is the result: {"translations": [{"id": 0, "text": "hello"}, {"id": 1, "text": "world"},]}"#,
            "{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"",
            r#"{"translations":[{"id":"0","text":"hello"},{"id":"1","text":"world"}]}"#,
        ] {
            let output = translations("test", response, &source).unwrap();
            assert_eq!(output.segments, expected);
            assert!(output.untranslated.is_empty(), "{output:?}");
        }
    }

    #[test]
    fn restores_input_order_from_ids() {
        let source = ["one".to_owned(), "two".to_owned()];
        let response = r#"{"translations":[{"id":1,"text":"world"},{"id":0,"text":"hello"}]}"#;
        assert_eq!(
            translations("test", response, &source).unwrap().segments,
            ["hello", "world"]
        );
    }

    #[test]
    fn tolerates_duplicate_missing_and_out_of_range_ids() {
        let source = ["one".to_owned(), "two".to_owned()];
        let short = r#"{"translations":[{"id":1,"text":"world"}]}"#;
        assert_eq!(
            translations("test", short, &source).unwrap().segments,
            ["one", "world"]
        );

        let response = concat!(
            r#"{"translations":["#,
            r#"{"id":0,"text":"hello"},"#,
            r#"{"id":0,"text":"duplicate"},"#,
            r#"{"id":9,"text":"extra"}"#,
            "]}"
        );
        assert_eq!(
            translations("test", response, &source).unwrap().segments,
            ["hello", "two"]
        );
    }

    /// The bug this reporting exists for: a reply cut off by the token cap
    /// parses, is the right length, and renders the tail in Japanese.
    #[test]
    fn a_truncated_reply_names_the_ids_it_never_reached() {
        let source = ["いち".to_owned(), "に".to_owned(), "さん".to_owned()];
        // Exactly what `max_tokens` produces: the array stops mid-entry, and
        // `json::from_str` repairs it into a valid two-item list.
        let cut_off = r#"{"translations":[{"id":0,"text":"one"},{"id":1,"text":"tw"#;
        let output = translations("test", cut_off, &source).unwrap();
        assert_eq!(output.segments, ["one", "tw", "さん"]);
        assert_eq!(output.untranslated, [2]);
        // Nothing here can know *why*; only the local backend sees a finish
        // reason, and it sets this afterwards.
        assert!(!output.truncated);
    }

    /// The shape of truncation that used to kill the whole page.
    ///
    /// The test above cuts inside a text *string*, which the repair closes into
    /// a complete entry. Cut two characters earlier -- anywhere in a segment's
    /// scaffolding -- and the repair produces `{"id":1}` or
    /// `{"id":1,"text":null}` instead, which `TranslationOutputSegment` cannot
    /// read. That failure used to propagate out of the `Vec` and fail the entire
    /// reply, so a page whose first bubbles translated perfectly rendered
    /// *wholly* untranslated. Where the model stopped decided which.
    ///
    /// Every one of these must now degrade exactly like the case above: the
    /// readable segments survive and the unreadable one is an ordinary miss.
    #[test]
    fn a_cut_in_the_scaffolding_loses_one_segment_rather_than_the_page() {
        let source = ["いち".to_owned(), "に".to_owned(), "さん".to_owned()];
        for cut_off in [
            r#"{"translations":[{"id":0,"text":"one"},{"#,
            r#"{"translations":[{"id":0,"text":"one"},{"id"#,
            r#"{"translations":[{"id":0,"text":"one"},{"id":"#,
            r#"{"translations":[{"id":0,"text":"one"},{"id":1"#,
            r#"{"translations":[{"id":0,"text":"one"},{"id":1,"#,
            r#"{"translations":[{"id":0,"text":"one"},{"id":1,"text"#,
            r#"{"translations":[{"id":0,"text":"one"},{"id":1,"text":"#,
        ] {
            let output = translations("test", cut_off, &source)
                .unwrap_or_else(|error| panic!("{cut_off:?} failed the page: {error}"));
            assert_eq!(
                output.segments[0], "one",
                "{cut_off:?} lost a segment that had already arrived"
            );
            // 1 and 2 keep their source and are reported, which is the whole of
            // the designed behaviour for an overflowing page.
            assert_eq!(output.untranslated, [1, 2], "{cut_off:?}");
        }
    }

    /// The forgiveness is confined to unreadable entries. A reply that is merely
    /// *wrong* -- an id out of range, a duplicate -- still goes through the
    /// existing rules rather than being silently dropped by the new one, and a
    /// reply that is not a translation list at all is still an error.
    #[test]
    fn a_readable_but_wrong_entry_is_not_quietly_discarded() {
        let source = ["いち".to_owned(), "に".to_owned()];
        let output = translations(
            "test",
            r#"{"translations":[{"id":0,"text":"one"},{"id":9,"text":"nine"}]}"#,
            &source,
        )
        .unwrap();
        assert_eq!(output.segments, ["one", "に"]);
        assert_eq!(output.untranslated, [1]);

        assert!(translations("test", r#"{"nonsense":true}"#, &source).is_err());
    }

    /// And the half of that bug the miss list cannot see at all: when the cut
    /// lands *inside* the final segment's text rather than before it, the
    /// repair fills every id, so `untranslated` is empty and the page looks
    /// clean from here. Only the backend's finish reason distinguishes it --
    /// which is why the pipeline stage must not gate its report on this list.
    #[test]
    fn a_cut_inside_the_last_segment_leaves_no_missed_ids_at_all() {
        let source = ["いち".to_owned(), "に".to_owned()];
        let cut_off =
            r#"{"translations":[{"id":0,"text":"one"},{"id":1,"text":"you should go to the sta"#;
        let output = translations("test", cut_off, &source).unwrap();
        assert_eq!(output.segments, ["one", "you should go to the sta"]);
        assert!(output.untranslated.is_empty(), "{output:?}");
        assert!(!output.truncated);
    }

    /// The name is the finding, and it is about the *miss list* only: every kind
    /// of miss still lands in `untranslated` identically, because that list is
    /// built from a filled-in mask and a mask has no room for a cause.
    ///
    /// That is exactly why `duplicate_ids` and `out_of_range_ids` exist beside
    /// it, and they are asserted here rather than in a test of their own so the
    /// two halves cannot drift apart: the same reply, reported one way that
    /// flattens the causes and another that keeps them.
    #[test]
    fn every_kind_of_miss_is_reported_the_same_way() {
        let source = ["いち".to_owned(), "に".to_owned(), "さん".to_owned()];
        // Id 0 answered twice, id 2 answered out of range, id 1 never sent.
        let response = concat!(
            r#"{"translations":["#,
            r#"{"id":0,"text":"one"},"#,
            r#"{"id":0,"text":"duplicate"},"#,
            r#"{"id":7,"text":"stray"}"#,
            "]}"
        );
        let output = translations("test", response, &source).unwrap();
        assert_eq!(output.segments, ["one", "に", "さん"]);
        assert_eq!(output.untranslated, [1, 2]);
        assert_eq!(output.duplicate_ids, 1);
        assert_eq!(output.out_of_range_ids, 1);
    }

    /// A clean reply must say so with zeroes, not merely fail to complain.
    ///
    /// Absent evidence and evidence of absence are the same value here -- both
    /// counters are plain `usize` -- so the only thing that keeps them honest is
    /// asserting the zero on a page that had nothing wrong with it.
    #[test]
    fn a_clean_reply_counts_no_dropped_ids() {
        let source = ["first line".to_owned(), "second line".to_owned()];
        let response =
            r#"{"translations":[{"id":0,"text":"the first"},{"id":1,"text":"the second"}]}"#;
        let output = translations("test", response, &source).unwrap();
        assert_eq!(output.segments, ["the first", "the second"]);
        assert!(output.untranslated.is_empty(), "{output:?}");
        assert_eq!(output.duplicate_ids, 0);
        assert_eq!(output.out_of_range_ids, 0);
    }

    /// One id answered twice, and nothing else wrong with the reply.
    ///
    /// Note what `untranslated` says on its own: `[2]`. Identical to a reply that
    /// simply stopped before the last segment, and the two want opposite fixes.
    #[test]
    fn a_repeated_id_is_counted_and_its_text_dropped() {
        let source = [
            "first line".to_owned(),
            "second line".to_owned(),
            "third line".to_owned(),
        ];
        let response = concat!(
            r#"{"translations":["#,
            r#"{"id":0,"text":"the first"},"#,
            r#"{"id":1,"text":"the second"},"#,
            r#"{"id":1,"text":"the second again"}"#,
            "]}"
        );
        let output = translations("test", response, &source).unwrap();
        // The FIRST answer wins and the repeat is discarded, rather than the
        // later one overwriting it. Anything else would make the rendered page
        // depend on reply order, which the prompt explicitly does not constrain.
        assert_eq!(output.segments, ["the first", "the second", "third line"]);
        assert_eq!(output.untranslated, [2]);
        assert_eq!(output.duplicate_ids, 1);
        assert_eq!(output.out_of_range_ids, 0);
    }

    #[test]
    fn an_id_past_the_end_is_counted_apart_from_a_duplicate() {
        let source = ["first line".to_owned(), "second line".to_owned()];
        let response = concat!(
            r#"{"translations":["#,
            r#"{"id":0,"text":"the first"},"#,
            r#"{"id":4,"text":"a segment that was never sent"}"#,
            "]}"
        );
        let output = translations("test", response, &source).unwrap();
        assert_eq!(output.segments, ["the first", "second line"]);
        assert_eq!(output.untranslated, [1]);
        assert_eq!(output.out_of_range_ids, 1);
        // Not folded into the other counter: under the local grammar the id is
        // schema-constrained to the page's range, so this one accuses the
        // grammar while a duplicate accuses the model.
        assert_eq!(output.duplicate_ids, 0);
    }

    #[test]
    fn both_kinds_at_once_are_counted_separately() {
        let source = [
            "first line".to_owned(),
            "second line".to_owned(),
            "third line".to_owned(),
        ];
        let response = concat!(
            r#"{"translations":["#,
            r#"{"id":0,"text":"the first"},"#,
            r#"{"id":0,"text":"the first once more"},"#,
            r#"{"id":0,"text":"and again"},"#,
            r#"{"id":9,"text":"nowhere"},"#,
            r#"{"id":3,"text":"also nowhere"}"#,
            "]}"
        );
        let output = translations("test", response, &source).unwrap();
        assert_eq!(output.segments, ["the first", "second line", "third line"]);
        assert_eq!(output.untranslated, [1, 2]);
        assert_eq!(output.duplicate_ids, 2);
        assert_eq!(output.out_of_range_ids, 2);
    }

    /// The accounting boundary, and the reason the loop is three arms rather
    /// than one condition with a counter bolted on.
    ///
    /// `out_of_order` and `placed` deliberately count only ids that were
    /// actually placed (see the comment above the loop). A duplicate or a stray
    /// id must therefore leave `previous` alone: counted as an ordinary entry it
    /// would register as disorder it did not cause, and the shuffle rate decides
    /// whether a positional reply schema -- no ids at all, ~1.2s of a ~4.4s page
    /// -- is safe to build.
    ///
    /// Asserted through `placed` on the debug log's own terms: the reply below
    /// is in strict ascending order for every id it actually places, with a
    /// low-numbered duplicate and a stray wedged between them where a naive
    /// counter would see two backward steps.
    #[test]
    fn a_duplicate_does_not_register_as_disorder() {
        let source = [
            "first line".to_owned(),
            "second line".to_owned(),
            "third line".to_owned(),
        ];
        let ordered = concat!(
            r#"{"translations":["#,
            r#"{"id":0,"text":"the first"},"#,
            r#"{"id":1,"text":"the second"},"#,
            r#"{"id":0,"text":"the first again"},"#,
            r#"{"id":8,"text":"nowhere"},"#,
            r#"{"id":2,"text":"the third"}"#,
            "]}"
        );
        let output = translations("test", ordered, &source).unwrap();
        // Every id placed, in order, despite the two backward-looking entries.
        assert_eq!(output.segments, ["the first", "the second", "the third"]);
        assert!(output.untranslated.is_empty(), "{output:?}");
        assert_eq!(output.duplicate_ids, 1);
        assert_eq!(output.out_of_range_ids, 1);

        /* The same three placements written without the noise. If a duplicate
         * or a stray perturbed `previous` or `placed`, this reply and the one
         * above would no longer be the same page from the ordering counter's
         * point of view -- which is the whole invariant, and the only way to
         * pin it from outside is that they agree on everything observable. */
        let clean = concat!(
            r#"{"translations":["#,
            r#"{"id":0,"text":"the first"},"#,
            r#"{"id":1,"text":"the second"},"#,
            r#"{"id":2,"text":"the third"}"#,
            "]}"
        );
        let baseline = translations("test", clean, &source).unwrap();
        assert_eq!(output.segments, baseline.segments);
        assert_eq!(output.untranslated, baseline.untranslated);

        /* And the counters themselves, which no return value carries. `placed`
         * is 3 and not 5: neither the duplicate nor the stray was placed. The
         * absence of an `out_of_order` field is the assertion for the other half
         * -- that counter is logged only on the branch taken when it is non-zero,
         * so the ordered branch firing IS the zero. */
        let logged = fields_logged_by(ordered, &source);
        assert_eq!(field(&logged, "placed"), Some(3), "{logged:?}");
        assert_eq!(field(&logged, "out_of_order"), None, "{logged:?}");
        // The new counters ride the same events, so the log says as much as the
        // return value does.
        assert_eq!(field(&logged, "duplicate_ids"), Some(1), "{logged:?}");
        assert_eq!(field(&logged, "out_of_range_ids"), Some(1), "{logged:?}");
    }

    /// The other side of that assertion: the ordering counter still fires when
    /// the reply really is shuffled. Without this, the test above would pass
    /// against a build where `out_of_order` had been deleted outright.
    #[test]
    fn a_genuinely_shuffled_reply_still_counts_as_disorder() {
        let source = ["first line".to_owned(), "second line".to_owned()];
        let shuffled =
            r#"{"translations":[{"id":1,"text":"the second"},{"id":0,"text":"the first"}]}"#;
        let logged = fields_logged_by(shuffled, &source);
        assert_eq!(field(&logged, "out_of_order"), Some(1), "{logged:?}");
        assert_eq!(field(&logged, "placed"), Some(2), "{logged:?}");
    }

    /// Every unsigned field of every `tracing` event one call emits, in order.
    ///
    /// `out_of_order` and `placed` are loop locals that are logged and thrown
    /// away -- by the time `Translated` exists the ids have been put back in
    /// order, so a shuffled reply and an ordered one are byte-identical. Scraping
    /// the event is the only way to see them, and it is what `replay.rs` does in
    /// production for exactly this reason.
    fn fields_logged_by(response: &str, source: &[String]) -> Vec<(String, u64)> {
        use tracing_subscriber::layer::SubscriberExt;

        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(FieldProbe(seen.clone()));
        tracing::subscriber::with_default(subscriber, || {
            translations("test", response, source).expect("the reply should parse");
        });
        let logged = seen.lock().expect("the probe should not have panicked").clone();
        logged
    }

    /// The first value logged under `name`, or `None` if no event carried it.
    ///
    /// Absence is a real answer here rather than a lookup failure: the ordering
    /// counter is logged only on the branch taken when it is non-zero.
    fn field(logged: &[(String, u64)], name: &str) -> Option<u64> {
        logged
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| *value)
    }

    struct FieldProbe(std::sync::Arc<std::sync::Mutex<Vec<(String, u64)>>>);

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for FieldProbe {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _: tracing_subscriber::layer::Context<'_, S>,
        ) {
            event.record(&mut FieldVisitor(&self.0));
        }
    }

    struct FieldVisitor<'a>(&'a std::sync::Mutex<Vec<(String, u64)>>);

    impl tracing::field::Visit for FieldVisitor<'_> {
        fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
            if let Ok(mut seen) = self.0.lock() {
                seen.push((field.name().to_owned(), value));
            }
        }

        // Required by the trait; every field this asserts on is a `usize`, which
        // `tracing` records through `record_u64` above.
        fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
    }

    #[test]
    fn a_reply_that_answers_nothing_reports_every_id() {
        let source = ["いち".to_owned(), "に".to_owned()];
        let output = translations("test", r#"{"translations":[]}"#, &source).unwrap();
        assert_eq!(output.segments, source);
        assert_eq!(output.untranslated, [0, 1]);
    }

    #[test]
    fn prompt_payload_contains_ordered_context() {
        let request = TranslationRequest::new(["new"], Language::English)
            .with_context([TranslationContext::new("old", "previous")]);
        let (_, user) = prompts(&request).unwrap();
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();
        assert_eq!(input["context"][0]["source"], "old");
        assert_eq!(input["context"][0]["translation"], "previous");
        assert_eq!(input["segments"][0]["id"], 0);
        assert_eq!(input["segments"][0]["text"], "new");
    }

    /// Both halves from the one call the pipeline makes -- `prompts` -- so the
    /// payload field and the system sentence cannot pass while the composed
    /// path ships unwired.
    #[test]
    fn a_glossary_reaches_both_the_payload_and_the_system_prompt() {
        let request = TranslationRequest::new(["new"], Language::English).with_glossary([
            TranslationContext::new("청풍검대", "the Clear Wind Swords"),
            TranslationContext::new("흑월문", "the Black Moon Sect"),
        ]);
        let (system, user) = prompts(&request).unwrap();
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();
        assert_eq!(input["glossary"][0]["source"], "청풍검대");
        assert_eq!(input["glossary"][0]["translation"], "the Clear Wind Swords");
        assert_eq!(input["glossary"][1]["source"], "흑월문");
        assert_eq!(input["glossary"][1]["translation"], "the Black Moon Sect");
        assert!(system.contains("glossary source term exactly as its given translation"));
    }

    /// A glossary-less page must read byte-identically to one built before the
    /// field existed: no `glossary` key in the payload, no sentence in the
    /// prompt. The whole-string baseline test above pins the prompt half too;
    /// this pins the payload half, which that test cannot see.
    #[test]
    fn an_empty_glossary_leaves_payload_and_prompt_without_a_trace() {
        let request = TranslationRequest::new(["new"], Language::English);
        let (system, user) = prompts(&request).unwrap();
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();
        assert!(input.get("glossary").is_none());
        assert!(!system.contains("glossary"));
    }

    #[test]
    fn system_prompt_encodes_invariants_and_custom_instructions() {
        let request = TranslationRequest::new(["hello"], Language::Korean)
            .with_source_language(Language::Japanese)
            .with_instructions("Use informal speech.");
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("from Japanese into natural Korean"));
        assert!(prompt.contains("Copy every input ID exactly once"));
        assert!(prompt.contains("Use informal speech."));
    }

    /// The whole OFF arm as one literal, deliberately not assembled from the
    /// code under test.
    ///
    /// Both levers default off, so this is the prompt every page goes through,
    /// and it is the baseline the other two arms are measured against. A drift of
    /// a single character turns a two-arm A/B into a three-way comparison in which
    /// the control also moved, and nothing downstream would say so -- the prompt is
    /// never recorded beside a page. `contains` assertions cannot catch that; only
    /// the whole string can.
    ///
    /// **CHANGED when the markup clause was added**, and this test's own warning
    /// is the reason the change is recorded here rather than made quietly:
    /// **every measurement taken before that change used a different control
    /// prompt.** Do not compare a translation number across that
    /// boundary without saying so. The name no longer claims "the prompt it always
    /// has", because it is not.
    #[test]
    fn an_unnamed_source_renders_the_current_baseline_prompt() {
        let request = TranslationRequest::new(["こんにちは"], Language::English);
        assert_eq!(
            translation_system_prompt(&request),
            concat!(
                "You are a professional manga translator. ",
                "Translate every input segment from the detected source language into natural ",
                "English. ",
                "Preserve character voice, emotional tone, relationship nuance, emphasis, and ",
                "sound effects while keeping wording concise enough for speech bubbles. ",
                "Each input segment has a numeric `id`. Return only a JSON object whose ",
                "`translations` array contains one object with `id` and translated `text` for ",
                "every input segment. Copy every input ID exactly once; order does not matter. ",
                "Never merge, split, omit, or add segments.",
                " Translated `text` must contain no LaTeX, Markdown, or other markup: ",
                "write every symbol as the literal character it is.",
                // CHANGED AGAIN when the quote steering was appended, so
                // measurements before that change used a different control
                // prompt. Same boundary rule as above.
                " When the translation needs quotation marks or apostrophes, write ",
                "the typographic characters “ ” ‘ ’ and never the ASCII ",
                "straight forms."
            )
        );
    }

    /// **The clause must be in EVERY arm, not just the default one.**
    ///
    /// The prompt has three levers -- `style_clause`, `name_medium`, and a named
    /// source language -- and the markup rule is a correctness requirement rather
    /// than an experimental arm, so no combination of them may drop it. Asserting
    /// it on the default arm alone would leave the replay binary's other arms
    /// emitting LaTeX with every test green.
    ///
    /// It also asserts the clause carries **no backslash of its own**: an example
    /// of the wrong form, sitting in the only text the model reads before it
    /// invents one, is an exemplar of the wrong form.
    #[test]
    fn the_markup_prohibition_survives_every_prompt_lever() {
        const CLAUSE: &str = "must contain no LaTeX, Markdown, or other markup";
        let arms = [
            TranslationRequest::new(["…"], Language::English),
            TranslationRequest::new(["…"], Language::English)
                .with_source_language(Language::Japanese),
            TranslationRequest::new(["…"], Language::Korean)
                .with_source_language(Language::Japanese)
                .with_instructions("Use informal speech."),
            TranslationRequest::new(["…"], Language::English)
                .with_context([TranslationContext::new("old", "previous")]),
        ];
        for (index, request) in arms.iter().enumerate() {
            let prompt = translation_system_prompt(request);
            assert!(prompt.contains(CLAUSE), "arm {index} dropped the markup rule");
            assert!(
                !prompt.contains('\\'),
                "arm {index} shows the model a backslash, which is the thing being forbidden"
            );
        }
    }

    /// The same contract as the markup rule above: the
    /// typographic-quotes instruction survives every prompt lever, and the
    /// prompt itself never SHOWS the model an ASCII straight quote -- the
    /// wrong form must not appear as an exemplar.
    #[test]
    fn the_quote_steering_survives_every_prompt_lever() {
        const CLAUSE: &str =
            "write the typographic characters “ ” ‘ ’ and never the ASCII straight forms";
        let arms = [
            TranslationRequest::new(["…"], Language::English),
            TranslationRequest::new(["…"], Language::English)
                .with_source_language(Language::Japanese),
            TranslationRequest::new(["…"], Language::Korean)
                .with_source_language(Language::Japanese)
                .with_instructions("Use informal speech."),
            TranslationRequest::new(["…"], Language::English)
                .with_context([TranslationContext::new("old", "previous")]),
        ];
        for (index, request) in arms.iter().enumerate() {
            let prompt = translation_system_prompt(request);
            assert!(prompt.contains(CLAUSE), "arm {index} dropped the quote steering");
            assert!(
                !prompt.contains('"'),
                "arm {index} shows the model a straight quote, which is the thing being steered away from"
            );
        }
    }

    /// OFF is the exact baseline -- the golden test above must
    /// not move -- and ON appends exactly one sentence, nothing else. The
    /// append property is an equality, the same strength as
    /// `naming_the_source_replaces_exactly_one_phrase`: a bare `contains`
    /// would stay green if the sentence also disturbed a neighbouring rule.
    #[test]
    fn the_containment_clause_appends_exactly_one_sentence_and_defaults_off() {
        const CLAUSE: &str = concat!(
            " When consecutive input segments form one continuing sentence, translate each ",
            "segment so its `text` covers only that segment's own words: do not move a clause ",
            "into a neighbouring segment to improve the English flow, and prefer a slightly ",
            "awkward split over redistributed content."
        );
        let off = translation_system_prompt(&TranslationRequest::new(["…"], Language::English));
        let on = translation_system_prompt(
            &TranslationRequest::new(["…"], Language::English).with_containment_clause(true),
        );
        assert!(
            !off.contains("own words"),
            "the baseline arm must not carry the clause -- OFF is the shipped default"
        );
        assert_eq!(on, format!("{off}{CLAUSE}"));
    }

    /// Same contract as the markup and quote rules above for the arms that can
    /// co-fire with it -- plus the one assertion those two do not need: the
    /// clause must land OUTSIDE the style window `style_clause_of` extracts
    /// (everything between the task sentence and `Each input segment`), or it
    /// silently becomes a fifth style arm and every style measurement stops
    /// being attributable. That is the trap the insertion-site comment
    /// documents; this is the test that keeps it a trap nobody springs.
    #[test]
    fn the_containment_clause_survives_every_prompt_lever_and_stays_out_of_the_style_window() {
        const MARKER: &str = "only that segment's own words";
        let arms = [
            TranslationRequest::new(["…"], Language::English).with_containment_clause(true),
            TranslationRequest::new(["…"], Language::English)
                .with_containment_clause(true)
                .with_source_language(Language::Japanese),
            TranslationRequest::new(["…"], Language::English)
                .with_containment_clause(true)
                .with_source_language(Language::Japanese)
                .with_instructions("Use informal speech."),
            TranslationRequest::new(["…"], Language::English)
                .with_containment_clause(true)
                .with_context([TranslationContext::new("old", "previous")])
                .with_glossary([TranslationContext::new("星盾", "Star Shield")]),
        ];
        for (index, request) in arms.iter().enumerate() {
            let prompt = translation_system_prompt(request);
            assert!(
                prompt.contains(MARKER),
                "arm {index} dropped the containment clause"
            );
            assert!(
                !style_clause_of(&prompt).contains(MARKER),
                "arm {index} let the clause into the style window"
            );
        }
    }

    /// NAMED is OFF with one noun phrase replaced, and nothing else.
    #[test]
    fn naming_the_source_replaces_exactly_one_phrase() {
        let off = translation_system_prompt(&TranslationRequest::new(["…"], Language::English));
        let named = translation_system_prompt(
            &TranslationRequest::new(["…"], Language::English)
                .with_source_language(Language::Japanese),
        );
        assert_eq!(
            named,
            off.replace("the detected source language", "Japanese")
        );
        assert!(named.contains("from Japanese into natural English"));
    }

    /// A bare `zh` is the only Chinese tag anything upstream produces, and it
    /// parses to *Simplified* Chinese. Asserting the source page is simplified is
    /// a claim about pixels nobody checked, so the prompt says "Chinese".
    ///
    /// The target is the opposite case and must keep its precision: `zh-CN` there
    /// is a caller asking for simplified output on purpose.
    #[test]
    fn a_bare_zh_source_is_named_chinese_and_the_target_is_not_coarsened() {
        let prompt = translation_system_prompt(
            &TranslationRequest::new(["你好"], Language::English)
                .with_source_language("zh".parse().unwrap()),
        );
        assert!(prompt.contains("from Chinese into natural English"), "{prompt}");
        assert!(!prompt.contains("Simplified Chinese"), "{prompt}");

        let into_chinese = translation_system_prompt(
            &TranslationRequest::new(["hello"], Language::ChineseSimplified)
                .with_source_language(Language::Japanese),
        );
        assert!(
            into_chinese.contains("into natural Simplified Chinese"),
            "{into_chinese}"
        );
    }

    /// MEDIUM names the medium, and never as a bare noun where the bare noun is
    /// the confusable one: "manhwa" and "manhua" differ by a letter, so each
    /// arrives beside the country it belongs to.
    #[test]
    fn the_medium_arm_pairs_the_noun_with_its_language() {
        for (language, expected) in [
            (Language::Korean, "You are a professional Korean manhwa translator."),
            (
                Language::ChineseSimplified,
                "You are a professional Chinese manhua translator.",
            ),
            (
                Language::ChineseTraditional,
                "You are a professional Chinese manhua translator.",
            ),
        ] {
            let prompt = translation_system_prompt(
                &TranslationRequest::new(["…"], Language::English)
                    .with_source_language(language)
                    .with_medium_term(true),
            );
            assert!(prompt.starts_with(expected), "{prompt}");
        }
    }

    /// The arm is a deliberate no-op on Japanese, and that is the finding rather
    /// than a gap: the unqualified prompt already says "manga", so a Japanese
    /// page has nothing to correct. Everything this arm can buy lies on the
    /// Korean and Chinese sources.
    #[test]
    fn the_medium_arm_changes_nothing_for_japanese() {
        let named = TranslationRequest::new(["…"], Language::English)
            .with_source_language(Language::Japanese);
        let medium = named.clone().with_medium_term(true);
        assert_eq!(
            translation_system_prompt(&medium),
            translation_system_prompt(&named)
        );
    }

    /// Without a source language there is nothing to derive a noun from, so the
    /// medium lever alone must not quietly become a fourth arm.
    #[test]
    fn the_medium_arm_alone_is_the_unnamed_prompt() {
        let request = TranslationRequest::new(["…"], Language::English).with_medium_term(true);
        assert_eq!(
            translation_system_prompt(&request),
            translation_system_prompt(&TranslationRequest::new(["…"], Language::English))
        );
    }

    /// Everything from "Preserve character voice" on is medium-neutral -- the
    /// JSON contract, the id rules, the bubble-length instruction -- so all three
    /// arms must share it verbatim. If an arm ever moved that text too, its
    /// effect would no longer be attributable to naming the language.
    #[test]
    fn every_arm_keeps_the_medium_neutral_tail() {
        let base = TranslationRequest::new(["…"], Language::English);
        let arms = [
            base.clone(),
            base.clone().with_source_language(Language::Korean),
            base.with_source_language(Language::Korean)
                .with_medium_term(true),
        ];
        let tails = arms
            .iter()
            .map(|request| {
                translation_system_prompt(request)
                    .split_once("Preserve character voice")
                    .expect("every arm keeps the neutral tail")
                    .1
                    .to_owned()
            })
            .collect::<Vec<_>>();
        assert_eq!(tails[0], tails[1]);
        assert_eq!(tails[1], tails[2]);
    }

    /// The style sentence alone, cut out of a rendered prompt by its two fixed
    /// neighbours rather than by matching any of its own words.
    ///
    /// That is what lets one helper read all four arms including the empty one:
    /// a `split_once("Preserve character voice")` -- which is how the tail test
    /// above works -- would panic on [`StyleClause::None`] rather than return the
    /// empty string it must return.
    fn style_clause_of(prompt: &str) -> String {
        let after = prompt
            .split_once("into natural English. ")
            .expect("the task sentence")
            .1;
        after
            .split_once("Each input segment has a numeric")
            .expect("the JSON contract")
            .0
            .to_owned()
    }

    const EVERY_STYLE_ARM: [StyleClause; 4] = [
        StyleClause::Shipped,
        StyleClause::NoSfx,
        StyleClause::None,
        StyleClause::Sized,
    ];

    /// The default is the shipped sentence, and naming it explicitly is a no-op.
    ///
    /// `an_unnamed_source_renders_the_prompt_it_always_has` already pins the
    /// bytes; what this adds is that the *default value of the new field* is the
    /// one that produces them, so the flag cannot ship silently switched on.
    #[test]
    fn the_style_clause_defaults_to_the_shipped_sentence() {
        let bare = TranslationRequest::new(["こんにちは"], Language::English);
        assert_eq!(bare.style_clause, StyleClause::Shipped);
        assert_eq!(
            translation_system_prompt(&bare.clone().with_style_clause(StyleClause::Shipped)),
            translation_system_prompt(&bare)
        );
    }

    /// Each arm renders the sentence it claims, asserted as a whole literal
    /// rather than by `contains`, for the same reason the baseline prompt is:
    /// these strings are the experiment, and a drifted one is a silently
    /// relabelled arm.
    #[test]
    fn each_style_arm_renders_the_sentence_it_claims() {
        for (clause, expected) in [
            (
                StyleClause::Shipped,
                "Preserve character voice, emotional tone, relationship nuance, emphasis, and \
                 sound effects while keeping wording concise enough for speech bubbles. ",
            ),
            (
                StyleClause::NoSfx,
                "Preserve character voice, emotional tone, relationship nuance, and emphasis \
                 while keeping wording concise enough for speech bubbles. ",
            ),
            (StyleClause::None, ""),
            (
                StyleClause::Sized,
                "Preserve character voice, emotional tone, relationship nuance, emphasis, and \
                 sound effects while keeping each translation as short as the meaning allows: a \
                 speech balloon has room for roughly one short line, and a longer translation is \
                 lettered smaller. ",
            ),
        ] {
            let prompt = translation_system_prompt(
                &TranslationRequest::new(["…"], Language::English).with_style_clause(clause),
            );
            assert_eq!(style_clause_of(&prompt), expected, "{clause:?}");
        }
    }

    /// The two arms that keep half the sentence must keep that half *verbatim*,
    /// or neither result is attributable to the half it changed.
    #[test]
    fn each_partial_arm_shares_its_untouched_half_with_the_shipped_one() {
        let of = |clause| {
            style_clause_of(&translation_system_prompt(
                &TranslationRequest::new(["…"], Language::English).with_style_clause(clause),
            ))
        };
        let shipped = of(StyleClause::Shipped);
        let tail = "while keeping wording concise enough for speech bubbles. ";
        let head = "Preserve character voice, emotional tone, relationship nuance, emphasis, and \
                    sound effects ";

        // NoSfx: one item off the preserve list, the concision tail untouched.
        let no_sfx = of(StyleClause::NoSfx);
        assert!(shipped.ends_with(tail), "{shipped:?}");
        assert!(no_sfx.ends_with(tail), "{no_sfx:?}");
        assert!(!no_sfx.contains("sound effects"), "{no_sfx:?}");
        // Not a bare deletion: the Oxford "and" moves onto "emphasis" so the
        // sentence is still English.
        assert!(no_sfx.contains("nuance, and emphasis while"), "{no_sfx:?}");

        // Sized: the preserve list verbatim, only the concision tail replaced.
        let sized = of(StyleClause::Sized);
        assert!(shipped.starts_with(head), "{shipped:?}");
        assert!(sized.starts_with(head), "{sized:?}");
        assert!(!sized.contains(tail), "{sized:?}");
        // A count is the thing this arm exists to avoid, so assert its absence
        // rather than trusting the literal above to stay countless.
        assert!(
            !sized.chars().any(|c| c.is_ascii_digit()),
            "the sized arm must not name a number: {sized:?}"
        );
    }

    /// Deleting the sentence must delete its separator with it. The failure this
    /// guards is not cosmetic: a doubled space, or a missing one welding
    /// "English." to "Each", is a fifth prompt nobody chose, and it would be
    /// scored as the `none` arm.
    #[test]
    fn deleting_the_style_sentence_leaves_exactly_one_space() {
        let prompt = translation_system_prompt(
            &TranslationRequest::new(["…"], Language::English)
                .with_style_clause(StyleClause::None),
        );
        assert!(
            prompt.contains("into natural English. Each input segment has a numeric `id`."),
            "{prompt}"
        );
        assert!(!prompt.contains("  "), "doubled space: {prompt}");
        assert!(!prompt.contains("Preserve"), "{prompt}");
    }

    /// The composition guarantee, and the reason both flags can be measured in
    /// the same harness: for a fixed style arm the sentence is byte-identical
    /// across all three language arms, so nothing the language lever does can be
    /// charged to the style lever.
    #[test]
    fn the_style_clause_is_identical_across_the_three_language_arms() {
        for clause in EVERY_STYLE_ARM {
            let base = TranslationRequest::new(["…"], Language::English).with_style_clause(clause);
            let arms = [
                base.clone(),
                base.clone().with_source_language(Language::Korean),
                base.clone()
                    .with_source_language(Language::Korean)
                    .with_medium_term(true),
            ];
            let clauses = arms
                .iter()
                .map(|request| style_clause_of(&translation_system_prompt(request)))
                .collect::<Vec<_>>();
            assert_eq!(clauses[0], clauses[1], "{clause:?}");
            assert_eq!(clauses[1], clauses[2], "{clause:?}");
        }
    }

    /// And the other direction, over all twelve combinations: every one renders a
    /// whole prompt with no doubled space, no unsubstituted placeholder, and both
    /// fixed ends intact. Twelve is small enough to enumerate, and an arm that
    /// only ever ran beside the default of the other flag would be untested in
    /// exactly the configuration a sweep uses.
    #[test]
    fn the_two_prompt_levers_compose_over_all_twelve_combinations() {
        for clause in EVERY_STYLE_ARM {
            for (language, medium) in [
                (None, false),
                (Some(Language::Korean), false),
                (Some(Language::Korean), true),
            ] {
                let mut request =
                    TranslationRequest::new(["…"], Language::English).with_style_clause(clause);
                if let Some(language) = language {
                    request = request
                        .with_source_language(language)
                        .with_medium_term(medium);
                }
                let prompt = translation_system_prompt(&request);
                let what = format!("{clause:?}/{language:?}/{medium}");
                assert!(
                    prompt.starts_with("You are a professional "),
                    "{what}: {prompt}"
                );
                // The markup rule is appended after the segment rules, so the
                // segment rules are not last. Both are still asserted -- the
                // point of this check is that BOTH fixed ends survive every lever combination, and
                // relaxing it to only the new tail would stop noticing if the
                // segment contract went missing.
                assert!(
                    prompt.contains("Never merge, split, omit, or add segments."),
                    "{what}: {prompt}"
                );
                // The quote steering is appended after the markup rule. The
                // markup rule stays asserted by containment, the new sentence
                // by position.
                assert!(
                    prompt.contains("write every symbol as the literal character it is."),
                    "{what}: {prompt}"
                );
                assert!(
                    prompt.ends_with("and never the ASCII straight forms."),
                    "{what}: {prompt}"
                );
                assert!(prompt.contains("into natural English. "), "{what}: {prompt}");
                assert!(!prompt.contains("  "), "{what}: doubled space in {prompt}");
                assert!(!prompt.contains('{'), "{what}: {prompt}");
            }
        }
    }

    /// **The escape override is on the LOCAL schema and must never be on the
    /// shared one.**
    ///
    /// `output_schema` is also sent to Gemini as `response_json_schema` and to
    /// OpenAI-compatible providers inside `response_format`. `x-guidance` is not a
    /// JSON Schema keyword, so a provider that validates strictly would reject the
    /// whole request — a remote outage caused by a local-only decoding fix. The
    /// negative half of this test is the one that matters.
    #[test]
    fn the_escape_override_rides_only_on_the_local_schema() {
        let local = local_output_schema(3);
        assert_eq!(
            local["x-guidance"]["json_allowed_escapes"], LOCAL_ALLOWED_ESCAPES,
            "the local schema must carry the override"
        );
        assert!(
            output_schema(3).get("x-guidance").is_none(),
            "the shared schema must NOT: it goes to Gemini and to OpenAI-compatible \
             providers, which do not know this key"
        );
        // Everything else must be identical, or the local arm is measuring a
        // different contract from the remote one.
        let mut stripped = local.clone();
        stripped.as_object_mut().unwrap().remove("x-guidance");
        assert_eq!(stripped, output_schema(3), "the override is the ONLY difference");
    }

    /// The escape set itself, pinned against what it is FOR.
    ///
    /// A future edit that "tidies" this back toward the JSON default would silently
    /// reopen the LaTeX-eating defect, and nothing downstream would say so — the
    /// grammar is never recorded beside a page.
    #[test]
    fn the_allowed_escapes_drop_every_letter_that_begins_a_latex_command() {
        // The four that destroyed a command, or would: \rightarrow, \blacksquare,
        // \times/\text, \frac.
        for letter in ['r', 'b', 't', 'f'] {
            assert!(
                !LOCAL_ALLOWED_ESCAPES.contains(letter),
                "\\{letter} is still legal, so a command starting with it can still be eaten"
            );
        }
        // A line break is the one control character a caption needs: 2,265
        // translated fields in the test corpus carry one.
        assert!(LOCAL_ALLOWED_ESCAPES.contains('n'), "real line breaks must survive");
        // Dialogue contains quotes and backslashes, and \uXXXX is how the model
        // reaches any other character at all.
        for required in ['\\', '"', 'u'] {
            assert!(
                LOCAL_ALLOWED_ESCAPES.contains(required),
                "{required:?} is not optional"
            );
        }
    }

    #[test]
    fn schema_requires_the_expected_number_of_id_text_pairs() {
        let schema = output_schema(3);
        let translations = &schema["properties"]["translations"];
        assert_eq!(translations["minItems"], 3);
        assert_eq!(translations["maxItems"], 3);
        assert_eq!(translations["items"]["properties"]["id"]["minimum"], 0);
        assert_eq!(translations["items"]["properties"]["id"]["maximum"], 2);
        assert_eq!(translations["items"]["additionalProperties"], false);
        assert_eq!(schema["additionalProperties"], false);
    }

    /// A segment may not be answered with nothing.
    ///
    /// The entry COUNT was already pinned by the test above, and that is what
    /// made the hole hard to see: a model that emits the right number of entries
    /// and leaves one of their texts empty passes every other clause here. It is
    /// not a formatting nicety -- the erase mask is written during detection, so
    /// an empty answer is a region rubbed off the page and lettered with nothing.
    #[test]
    fn a_segment_may_not_be_answered_with_an_empty_string() {
        let schema = output_schema(3);
        let text = &schema["properties"]["translations"]["items"]["properties"]["text"];
        assert_eq!(text["type"], "string");
        assert_eq!(
            text["minLength"], 1,
            "an empty translation must not satisfy the grammar"
        );
    }

    #[test]
    fn empty_custom_instructions_are_ignored() {
        let request = TranslationRequest::new(["hello"], Language::English).with_instructions("  ");
        assert!(!translation_system_prompt(&request).contains("Additional instructions"));
    }

    #[test]
    fn context_is_reference_only() {
        let request = TranslationRequest::new(["Where is she?"], Language::Japanese)
            .with_context([TranslationContext::new("I saw Alice.", "アリスを見た。")]);
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("dialogue continuity"));
        assert!(prompt.contains("Do not translate or return the context"));
    }

    /// A `\u0000` escape is valid JSON, so this arrives through the STRICT
    /// serde_json path with nothing repaired -- the reply is otherwise perfect.
    /// Left alone it reaches `validate_authored_text` and fails the whole page.
    ///
    /// See `both_parse_paths_really_do_yield_a_nul` below for the mechanism
    /// pinned without reference to the fix.
    #[test]
    fn a_unicode_escaped_nul_is_stripped_from_a_translation() {
        let reply = r#"{"translations":[{"id":0,"text":"Wait\u0000 a sec!"}]}"#;
        let translated = translations("local", reply, &["まって！".to_owned()]).unwrap();
        assert_eq!(translated.segments, ["Wait a sec!"]);
        assert!(translated.untranslated.is_empty());
    }

    /// The other route: a raw NUL byte, which is what a byte-fallback token
    /// decodes to. Strict parsing rejects a control character inside a string,
    /// so this one only survives because `json.rs` repairs it -- and the repair
    /// parser pushes any non-escape character through verbatim.
    #[test]
    fn a_raw_nul_byte_is_stripped_too() {
        let reply = "{\"translations\":[{\"id\":0,\"text\":\"Good\0 morning!\"}]}";
        let translated = translations("local", reply, &["おはよう！".to_owned()]).unwrap();
        assert_eq!(translated.segments, ["Good morning!"]);
    }

    /// The segment must still count as translated. Dropping it instead would
    /// silently render the Japanese, which is the failure mode `untranslated`
    /// exists to make visible and is barely better than the crash.
    #[test]
    fn a_segment_that_was_only_a_nul_is_still_translated() {
        let reply = r#"{"translations":[{"id":0,"text":"\u0000"}]}"#;
        let translated = translations("local", reply, &["…".to_owned()]).unwrap();
        assert_eq!(translated.segments, [""]);
        assert!(translated.untranslated.is_empty());
    }

    /// The common path must be untouched, allocation aside: every ordinary page
    /// goes through this function for every segment.
    #[test]
    fn text_without_a_nul_is_returned_unchanged() {
        let reply = r#"{"translations":[{"id":0,"text":"Beautiful\nmorning — isn't it?"}]}"#;
        let translated = translations("local", reply, &["source".to_owned()]).unwrap();
        assert_eq!(translated.segments, ["Beautiful\nmorning — isn't it?"]);
    }

    /// The mechanism, pinned independently of the fix, so deleting `without_nul`
    /// does not also delete the evidence for why it exists. Both parse paths
    /// really do hand back a NUL; this is not a defence against a hypothetical.
    #[test]
    fn both_parse_paths_really_do_yield_a_nul() {
        let escaped: TranslationOutput =
            crate::json::from_str(r#"{"translations":[{"id":0,"text":"a\u0000b"}]}"#).unwrap();
        assert!(escaped.translations[0].text.contains('\0'), "escaped path");

        // Strict serde_json refuses a control character inside a string, so this
        // one only survives because the repairing parser accepts it verbatim.
        let raw: TranslationOutput =
            crate::json::from_str("{\"translations\":[{\"id\":0,\"text\":\"a\0b\"}]}").unwrap();
        assert!(raw.translations[0].text.contains('\0'), "raw path");
    }

    /// The guard the crash proves we need: whatever this function returns must be
    /// committable, so assert the scene's own rule over every segment rather than
    /// over the one the test happened to poison.
    #[test]
    fn no_segment_can_carry_a_nul_out_of_this_function() {
        let reply = r#"{"translations":[
            {"id":0,"text":"clean"},
            {"id":1,"text":"po\u0000isoned"},
            {"id":2,"text":"\u0000\u0000"}
        ]}"#;
        let sources = ["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let translated = translations("local", reply, &sources).unwrap();
        assert!(
            translated
                .segments
                .iter()
                .all(|segment| !segment.contains('\0')),
            "{:?}",
            translated.segments
        );
    }
}
