//! Command line, and the pure resolution of it into startup state.
//!
//! `resolve` takes the environment token as an argument rather than reading the
//! process environment, so every rule it enforces is testable.

use std::{net::SocketAddr, num::NonZeroU32, time::Duration};

use anyhow::{Result, anyhow, bail};
use clap::Parser;
use koharu_translator::{OpenAiCompatibleConfig, ProvidersConfig};
use url::Url;

use crate::{
    engine::{Defaults, Pinned},
    guard::{allowed_hosts_from, digest},
    models, vram,
};

const DEFAULT_MAX_UPLOAD_BYTES: usize = 32 * 1024 * 1024;

/// The most `--story-pairs` will accept.
///
/// Read off KV arithmetic computed for **`gemma4-26b-a4b-it`**, the default
/// translator -- see that flag's doc comment for the whole calculation, and for
/// what happens when a heavier `--llm` is pinned under a ceiling sized for this
/// one. It is not a round figure chosen for looks, and it is not model-agnostic.
const MAX_STORY_PAIRS: usize = 1024;

/* THE CEILING IS AFFORDABILITY, NOT ADVICE, AND 1024 HAS BEEN MEASURED AND
 * REJECTED. It was raised on the arithmetic above, and the very first thing run
 * under it says do not use it.
 *
 * `gemma4-26b-a4b-it` at temperature 0 over a Japanese test volume (213 pages of
 * one volume, 2,441 segments — the benchmark corpus CANNOT test this, because a
 * story accumulates within one source and its four sources cap at 209/113/94/95
 * pairs, so 96-vs-anything is a no-op on two of them):
 *
 *   window   corpus pass   unanswered   truncated   blind panel mean
 *   96       240,704 ms    7            2           8.27
 *   1024     731,778 ms    14           3           8.33
 *
 * The window BITES — 187 of the 194 pages that carried more than 96 pairs
 * produced different output, so this is not a null experiment. It buys nothing:
 * +0.06 on a 1-10 scale is 0.4 standard errors, the paired sign is 6 better / 6
 * worse / 0 equal, and FIDELITY IS EXACTLY TIED at 8.33 — which is the criterion
 * the story window exists to buy. For **3.04x** the translator wall clock, on the
 * model chosen for being 1.4-1.6x faster end to end.
 *
 * The doubling of unanswered segments is the finding worth keeping: 7 -> 14, a
 * direct observation of long-prompt degradation rather than a hypothesis about
 * it. A longer prompt does not merely cost time; it costs answers.
 *
 * Measured on JAPANESE ONLY. The test volume is one manga volume, and no strip
 * source measured carries enough segments to fill a window this size, so nothing
 * here speaks to Chinese or Korean.
 */

/// What the renderer's own `RenderTheme::default()` asks for.
///
/// "Arial" is not redundant. The resolver appends it anyway when it is absent
/// (`koharu-renderer/src/fonts.rs`), but naming it here makes the fallback
/// visible at the only place anyone reads to find out what the pages are set in.
const DEFAULT_FONT_FAMILIES: [&str; 2] = ["CCWildWords", "Arial"];

// The register instruction that was benchmarked and REJECTED lives in the
// `translation_instructions` doc comment below rather than in a const here: it
// is a reproduction recipe, not a value the binary uses. An unused const is a
// rustc warning, and zero rustc warnings from this crate is the bar.

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Local Koharu translation server for the BireLate extension"
)]
pub struct Cli {
    /// Loopback by default. A non-loopback bind requires a token.
    #[arg(long, default_value = "127.0.0.1:8765")]
    pub addr: SocketAddr,

    /// Shared secret the extension must send as X-Koharu-Token. Falls back to
    /// BIRELATE_TOKEN, then to a freshly generated one printed at startup.
    #[arg(long)]
    pub token: Option<String>,

    /// Serve without authentication. Any web page can then reach the server.
    #[arg(long, conflicts_with = "token")]
    pub no_token: bool,

    /// Run the pipeline on the CPU instead of the GPU (much slower).
    #[arg(long)]
    pub cpu: bool,

    // `manga-ocr` displaced `baberu-ocr` after a three-way benchmark on real
    // pages -- 66 regions over 6 pages of a real volume, each engine run twice.
    //
    // The decisive number: **manga-ocr and paddleocr-vl-1.6 disagree on kanji in
    // zero of 66 regions**, while baberu-ocr differs from *both* on the same
    // four. Two independent architectures agreeing perfectly is far stronger
    // evidence than a majority vote, and all four were then confirmed against
    // the printed page: baberu read four kanji as visually similar characters
    // the page does not print.
    // Each of those is a real word replaced by a non-word, which is the failure
    // that matters -- the LLM confabulates a plausible sentence from it and the
    // result reads like a bad translation rather than a bad read.
    //
    // baberu is *better* at punctuation (it kept a closing `』` manga-ocr
    // dropped, and a trailing clause manga-ocr lost). That is the trade, and it
    // is the right way round: punctuation slips survive translation, kanji
    // substitutions do not.
    //
    // manga-ocr is also the fastest by a wide margin -- 356ms/page against
    // baberu's 1023ms and PaddleOCR-VL's 2270ms, warm. It makes the *inpainting*
    // stage faster too (442ms vs 1140/2103), because a smaller OCR model leaves
    // enough headroom that admission control stops evicting the inpainter.
    //
    // `paddleocr-vl-1.6` then became the default, and `manga-ocr` is EARNED
    // rather than assumed: the extension picks it only
    // on positive evidence that the material is Japanese AND paged
    // (`extension/script-latch.js`). Everything else -- Chinese, Korean,
    // undeclared, a Japanese *webtoon*, or an unclassifiable page shape -- reads
    // with the bilingual engine.
    //
    // The asymmetry that chose the direction, and it is measured: guessing
    // PaddleOCR-VL wrongly costs ~2s a page, is loud and reversible, and showed
    // no separation on dialogue in a blind panel on a Japanese webtoon (15-9,
    // p = 0.31). Guessing manga-ocr wrongly costs SILENT FABRICATION -- on 10
    // boxes two blind seats called pure artwork, manga-ocr lettered 8 and paddle
    // 4, because manga-ocr answers unreadable ink with fluent kana that nothing
    // refuses on a Japanese page while paddle answers with a symbol the
    // punctuation gate already catches.
    //
    // This default and the extension's must match or the FIRST translate of
    // every session forces a `Pipeline::reload` -- see the test below.
    //
    // The default is now `hunyuan-ocr-1.5`, with PaddleOCR-VL demoted to the
    // reserve. PaddleOCR-VL stays wired and selectable; the paragraphs above
    // record why IT displaced manga-ocr in its day, and stay because the
    // fabrication asymmetry they measure is still what the reserve order rests
    // on. The default needs the HunyuanOCR sidecar (started by
    // `scripts/serve.ps1`) reachable at load, or OCR fails at start with one
    // clear message naming the endpoint.
    /// OCR engine: hunyuan-ocr-1.5, paddleocr-vl-1.6, manga-ocr, baberu-ocr or
    /// ollama-vision. The default needs the HunyuanOCR sidecar reachable at startup.
    #[arg(long, default_value = "hunyuan-ocr-1.5")]
    pub ocr: String,

    /// Serve every request for `hunyuan-ocr-1.5` -- the default above, and the
    /// extension's -- with ENGINE instead, for a machine without the HunyuanOCR
    /// sidecar. Responses and `/status` name ENGINE, the engine that ran. Any
    /// `--ocr` value except `hunyuan-ocr-1.5` itself.
    #[arg(long, value_name = "ENGINE")]
    pub hunyuan_substitute: Option<String>,

    /// Where Koharu keeps its downloaded runtimes and models. Falls back to
    /// BIRELATE_STORE_DIR, then to Koharu's own `%LOCALAPPDATA%\koharu\packages`.
    /// A relative path is taken from the current directory.
    #[arg(long, value_name = "PATH")]
    pub store_dir: Option<std::path::PathBuf>,

    /// Engine that erases the source text: lama, aot-inpainting, rorem-mixed or
    /// flux2-klein.
    #[arg(long, default_value = "lama")]
    pub inpainting: String,

    /// Which backend runs the translation LLM. Fixed for the process.
    #[arg(long, default_value = "local")]
    pub provider: String,

    /// Model name. A Koharu registry id for `--provider local`, an Ollama tag
    /// for `--provider ollama`. Fixed for the process.
    #[arg(long)]
    pub llm: Option<String>,

    /// Override the OpenAI-compatible endpoint. Defaults to Ollama's
    /// http://localhost:11434/v1.
    #[arg(long, value_name = "URL")]
    pub base_url: Option<Url>,

    /// Language to translate into, as a BCP 47 tag such as en-US.
    #[arg(long, default_value = "en-US")]
    pub target_language: String,

    // Appended verbatim to the LLM's system prompt, once per page.
    //
    // **Unset on purpose, and that is a measured decision rather than an
    // oversight.** A register instruction (naming manga dialogue, asking for
    // preserved honorifics, matching register, balloon-short lines, no
    // translator notes) was briefly the default. A blind three-judge panel over
    // the 39 regions where it changed the English scored it **34 / 31 / 52 tie**
    // against no instruction at all -- a coin flip, with the three judges
    // disagreeing on direction. Its own stated goal did not survive either:
    // honorifics and titles were dropped on *both* sides on different lines.
    //
    // So the flag stays reachable and the default stays empty. Do not restore a
    // default without re-measuring; "it obviously helps" is exactly the belief
    // the panel refuted. Note the null result is over one dialogue-heavy title,
    // so it bounds the effect on *this* kind of material, not on every kind.
    //
    // To reproduce the rejected experiment, pass:
    //
    // > This is manga dialogue. Use natural, colloquial English and keep each
    // > speaker's register -- casual speech stays casual, formal speech stays
    // > formal. Preserve honorifics and the original name order. Keep each line
    // > short enough to sit in a speech balloon. Do not add translator notes,
    // > explanations, or bracketed romanisation.
    //
    // Whatever you pass, do not name a source language *here*. The reasoning is
    // unchanged and still binding for this flag: no OCR backend reports a
    // language -- `SourceText.language` echoes the request's declaration, and
    // was once a hardcoded `ja-JP` -- so naming a language
    // here would be an assertion rather than a fact, and a real reading session
    // has already run against a Chinese manhua. This flag is page-invariant
    // config, so anything it says about the source is said about every page the
    // process ever sees, which is the worst possible place for a guess.
    //
    // **A source language is not only guessable.**
    // `TranslationRequest::with_source_language` is ungated and `prompt.rs`
    // renders three arms -- unnamed, named, and named
    // with the medium noun beside it -- so a caller that *knows* can say so
    // through the request rather than through this string. Two callers can
    // know: `replay --source-language-prompt off|named|medium` reads it from the
    // freeze manifest, which is where it is measurable today, and a future
    // per-request server field defaulting to auto would be the shipping route.
    // Neither is this flag, and neither is the pipeline: `stages/translation.rs`
    // still never calls it, so nothing a reader's page goes through has changed.
    //
    // It lives on `PipelineConfig`, so changing it between requests forces a
    // full pipeline reload -- hence a startup flag rather than a form field.
    /// Extra text appended verbatim to the translator's system prompt, once per page.
    /// Unset by default; fixed for the life of the process.
    #[arg(long)]
    pub translation_instructions: Option<String>,

    // Whether the translator's system prompt carries the containment
    // sentence: each segment's translation covers that segment's own source
    // and nothing of its neighbours', even where one sentence spans several
    // segments. **OFF by default** -- a changed system prompt resamples every
    // generation, so the arm ships selectable for a same-binary A/B rather
    // than on.
    //
    // The measured slide class it targets --
    // a region lettering its neighbour's clause -- is RE-PARTITIONING under
    // English head-initial word order, not missed adjacency: the model had
    // already joined every in-reach multi-segment sentence and then moved
    // the verb or negation up into the earlier segment. Marking
    // continuations instead was measured and rejected: the best
    // source-detectable rule fires on ~85% of pages at sub-1% defect
    // precision, and its false fires endorse the stammer-fabrication class.
    //
    // Lives on `PipelineConfig` like `--translation-instructions`, so it is
    // process-fixed; the wire reports it per page as `containment_clause`,
    // the same self-description contract as `story_context` and
    // `glossary_terms`.
    /// Ask the translator, in its system prompt, to keep each segment's translation to
    /// that segment's own source text. Default: off.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub containment_clause: Option<bool>,

    // Describe each segment to the translator: what KIND of text it is, and
    // whether the artwork strikes it through. **OFF by default.**
    //
    // The model has always been handed a bare list of strings -- `{id, text}`
    // and nothing else -- so a caption, a scream and a name the page visibly
    // CANCELS all arrived looking identical. On one test page the author strikes
    // a name through and writes the true name beside it; the glyphs are read
    // correctly and the reply was a romanisation that is neither the name's
    // meaning nor its sound. An invented fluent name, which every counter here
    // scores as a perfect translation.
    //
    // Both facts were already measured and both died before the translator:
    // the strike in `Typography.extensions` (read only by the compositor), the
    // kind in the scene's `Region`. This carries them the rest of the way and
    // tells the model what a struck name MEANS -- that its replacement is
    // elsewhere on the same page.
    //
    // **A prompt change re-rolls every page**, so the off arm is byte-exact by
    // construction: with this false the segment list serializes exactly as it
    // always has and the extra prompt sentence is never appended.
    /// Tell the translator what kind of text each segment is and whether the artwork
    /// strikes it through. Default: off.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub segment_context: Option<bool>,

    // Sampling temperature for the translation LLM.
    //
    // Unset means the model descriptor's own value, which for both Gemma-4
    // arms -- the default `gemma4-26b-a4b-it` and `gemma4-31b-it` -- is
    // temperature 1.0 / top_k 64 / top_p 0.95, Google's recommended
    // defaults for Gemma, not an accident. (They are identical, so switching
    // the default between them was not also a sampling change. Other
    // descriptors are not: `ministral-3-14b-instruct` carries 0.05.)
    // Lowering it is defensible for
    // translation, which wants faithfulness over variety, but it is a *guess*
    // until measured against the fixture, so the default deliberately changes
    // nothing. `GenerationConfig` was previously `::default()` at every call
    // site, which made this unreachable at any price.
    //
    // Note the seed is a fixed constant upstream, so an A/A pass has a zero
    // nondeterminism floor -- a temperature A/B is readable without averaging.
    /// Sampling temperature for the translation model. Unset uses the model's own
    /// default (1.0 for the Gemma-4 models).
    #[arg(long, value_name = "T")]
    pub translation_temperature: Option<f32>,

    // Whether the translator's sliding-window layers allocate the whole
    // context. Unset leaves llama.cpp's own default, which today is `true`.
    //
    // This is a VRAM lever and a large one, because gemma-4 is mostly a
    // sliding-window model: **50 of its 60 layers** attend within a 1024-token
    // window, and they carry **800 KiB of the 880 KiB per token** the KV cache
    // costs. With the full cache those 50 layers are sized at `n_ctx`; with the
    // reduced one they stop at about the window plus a ubatch (~1536 cells) and
    // never grow again.
    //
    // So the saving is zero below `n_ctx` ~1536 -- not negative, llama.cpp
    // clamps the reduced size against the base -- and linear above it: ~2.1 GiB
    // at the 4250 a dense page already reaches, ~5.2 GiB at 8192, ~11.6 GiB at
    // 16384. Measured at `--story-pairs` 96 it is **not** needed -- `n_ctx`
    // peaked at 3,840, or 3.22 GiB -- and it becomes the right call only for a
    // window several times larger again.
    //
    // **Every absolute figure above is `gemma4-31b-it`'s, and the default is
    // no longer that model.** The shape is unchanged for `gemma4-26b-a4b-it`
    // -- also mostly sliding-window, 25 of its 30 blocks -- but it carries
    // **220 KiB per token, not 880**, so divide each saving by about four. The
    // `n_ctx` crossover at ~1536 is a cell count and does not move.
    //
    // **Why it is safe here specifically.** The reduced cache discards state
    // once it leaves the window, so upstream warns it breaks context reuse,
    // prompt reuse, `llama_state_seq_*` save/restore and multi-sequence work.
    // koharu does none of the four: one `new_context` call site in the whole
    // tree, created per inference and dropped on return; one prompt decode then
    // one decode per generated token, each appending at a new maximal position;
    // `context/session.rs` and `context/kv_cache.rs` are dead API surface with
    // zero callers; `n_seq_max` is 1 and every sequence id is 0.
    //
    // **It is NOT guaranteed to produce identical tokens**, and the reason is
    // not the obvious one. The arithmetic is correct -- the `+ n_ubatch` term
    // covers the ubatch boundary, where the first query of a 512-token ubatch
    // still needs keys a window back. But sampling here is stochastic
    // (temperature 1.0 / top_k 64 / top_p 0.95), so token identity needs
    // bit-identical logits, and changing `n_kv` changes the CUDA attention
    // reduction blocking, which is not associativity-safe. Judge an A/B on
    // quality, never on a diff.
    //
    // Worth passing explicitly even to ask for the current behaviour: llama.cpp
    // is pinned as a hardcoded `const RELEASE` in koharu-runtime rather than in
    // a lockfile, so this default arrives from a binary that can change on an
    // ordinary rebase with nothing in a diff to notice.
    /// Whether the translator's sliding-window attention layers allocate a cache for
    /// the whole context. Unset keeps llama.cpp's default (true); false saves VRAM on
    /// long prompts.
    #[arg(long, value_name = "BOOL")]
    pub swa_full: Option<bool>,

    // `last-resort` (default), `normal`, `disabled`, or `auto` for the
    // renderer's own rule.
    //
    // Measured, three arms over 31 real pages. **Disabling hyphenation was
    // rejected**: a blind three-lens panel split 11-7 *against* it, because
    // never breaking forces the text smaller and two of three lenses weigh
    // legibility above craft. Only the letterer lens preferred it (5-1); the
    // reader and skeptic lenses went the other way (5-1 each).
    //
    // What both opposing lenses independently found instead is that hyphens
    // often **buy nothing** -- same measured size, same line count, hyphens
    // anyway, on four boxes across three pages. That is `normal` misfiring, and
    // it was the default for every layer except horizontal English inside a
    // bubble, so free-standing captions -- the narrowest boxes on a page -- got
    // the most aggressive setting of the three.
    //
    // `last-resort` cannot misfire by construction: it breaks only when the
    // unhyphenated text would overflow. Measured against the old behaviour it
    // costs **zero** extra overflow warnings (6 either way) and changes the
    // render on 9 of 31 pages, against 27 of 31 for `disabled`. A surgical
    // change where disabling was a blunt one.
    /// Hyphenation policy for the English lettering: last-resort (break only to avoid
    /// overflow), normal, disabled, or auto.
    #[arg(long, value_name = "POLICY", default_value = "last-resort")]
    pub hyphenation: String,

    // How far above a page's prevailing dialogue size one balloon may be set.
    // Anything below 1.0 turns the rule off and lets every balloon solve alone.
    //
    // Auto-fit is per-layer, so the size a balloon settles at encodes *its own
    // area* rather than the voice speaking. The 20-page render audit found this
    // on ~18 of 20 pages: 4:1 spreads inside a single page, and one continuous
    // sentence cut across two cells of the same bubble solved at 44px and 19px.
    // A reader reads a size change as emphasis, so incidental ones read as
    // shouting.
    //
    // Only an upper bound is available. Auto-fit already returns the largest
    // size that fits, so a small balloon is small because nothing larger fits
    // and no page rule can raise it; coherence is bought by bringing the
    // outliers down. The ceiling is never below the prevailing size itself,
    // which is what keeps the balloon fill.
    /// How far above a page's prevailing dialogue size one balloon may be lettered, as
    /// a ratio. Values below 1.0 turn the cap off.
    #[arg(long, value_name = "RATIO", default_value_t = 1.25)]
    pub size_coherence: f32,

    // How many source/target pairs a story carries into the next page's prompt.
    // `0` turns the window off, which is the A/B's other arm.
    //
    // **This was once a compile-time constant**, which is why the
    // window has only ever been measured at two sizes: switching arms cost a
    // rebuild. 96 is unchanged as the default.
    //
    // **What the window costs is measured, on manga: 22.9% of wall clock.** 40
    // pages, OCR byte-identical on 40 of 40 so that only the prompt differed --
    // page median 5,712 ms with a story against 4,746 ms without.
    //
    // **What it buys is naming, measured over a whole 213-page volume.** A main
    // character's name kept one romanisation across 31 renders with **one**
    // drift, against a variant spelling roughly 1 page in 11 with no story.
    //
    // **24 was measurably too small**, on a Chinese webtoon: the chapter's title
    // term fragmented into three renderings, because a slice carries ~0.7 pairs
    // and the window had saturated by slice 20 of 219.
    //
    // **The ceiling is 1,024, it is KV arithmetic rather than taste, and it is
    // a property of the DEFAULT MODEL rather than of this flag.** Raised from
    // 256 when `gemma4-26b-a4b-it` became the default.
    //
    // `context_values` sets `n_ctx = prompt + max_tokens + 1`, so a carried pair
    // is charged at the same per-token KV rate a generated one is. Measured: a
    // manga pair costs **21.5 tokens**, and at 96 pairs `n_ctx` peaked at
    // **3,840** -- which is 3.22 GiB of KV at the incumbent's 880 KiB per token
    // and **825 MiB** at this model's printed 220.
    //
    // **The old comment's arithmetic was wrong, and the correction is the
    // interesting part.** It weighed that peak against a ~7.9 GiB "translation
    // stage allowance" that charged NEITHER the other three stages' resident
    // weights NOR the compute plateau. Charged properly, on the figures koharu
    // printed for a 32 GB card:
    //
    // ```text
    //   30,193 MiB  budget
    // -  3,019 MiB  koharu's safety_reserve
    // -  3,052 MiB  detection + OCR + inpainting, resident
    // -    537 MiB  compute plateau
    // = 23,585 MiB  for the translator: its weights AND its KV
    // ```
    //
    // Subtract the weights and what is left is the KV a page may use before
    // admission control stops keeping the other stages resident. Against that
    // denominator, a 256-pair window costs:
    //
    // | model | weights | KV/token | 256 pairs |
    // |---|---|---|---|
    // | `gemma4-31b-it` | 16.09 GB | 880 KiB | **96%** on Japanese, **126%** on the densest Chinese |
    // | `gemma4-26b-a4b-it` | 13.26 GB | 220 KiB | **22%** |
    //
    // So **256 was never safe for the incumbent** -- over the line on the
    // Chinese material and within 4% of it on Japanese -- and it is about a
    // quarter of what the MoE carries comfortably. Measured, the MoE's headroom
    // runs out near **1,563** pairs on the worst source and **2,401** on manga.
    // 1,024 sits a factor of 1.5 below the first and 2.3 below the second, which
    // is the margin this ceiling is buying; it is deliberately not the largest
    // affordable number.
    //
    // **Breaching the headroom does not crash, it thrashes.** `admission_plan`
    // returns `unload_idle: true` and evicts every other stage's weights on
    // every request, measured at **5,189 -> 12,790 ms/page**. And it latches:
    // `Residency::workspace_bytes` only ever grows, so the single densest page
    // ever translated raises the threshold for every page after it, for the life
    // of the process.
    //
    // **This guard cannot protect a model it was not computed for, and that is
    // the thing to know before pinning `--llm`.** `--llm gemma4-31b-it
    // --story-pairs 2000` is refused, by this ceiling. `--llm gemma4-31b-it
    // --story-pairs 1000` is **accepted, and thrashes** -- roughly four times
    // the KV that model can afford, so every page pays the eviction above.
    // Pin a model heavier than the default and set its ceiling by hand -- for
    // the incumbent that is *below* 256, not at it, per the table above -- or
    // drop to `--swa-full false`, whose saving is linear in `n_ctx`.
    //
    // Over the ceiling the value is **refused rather than clamped**: a window
    // that silently became a different window is exactly the failure this flag
    // exists to make measurable.
    //
    // Note 21.5 tokens is the *manga* figure, and pairs are the budget while
    // pages are not: manga fills 11.28 pairs a page and a webtoon slice 0.726,
    // so the same 96 is ~8.5 pages of one and ~130 slices of the other.
    /// How many recent source/translation pairs from the same story are carried into
    /// the next page's prompt, up to 1024. 0 turns the story window off.
    #[arg(long, value_name = "PAIRS", default_value_t = crate::story::DEFAULT_PAIRS)]
    pub story_pairs: usize,

    // Keep a sound effect's pair OUT of the story window, so "Boom!" is never
    // carried forward as established terminology. **ON by default**, on a
    // full-chapter A/B.
    //
    // The window taught this defect, measured over 54 archived renders of one
    // Chinese test page: with the window OFF an ability name ending in a
    // repeated `爆爆爆` is kept 40 of 40 times; with it ON it collapses to
    // "Boom! Boom! Boom! Boom!" in 12 of 14 -- and the same inversion
    // reproduces on another corpus in another language, where a name built on a
    // repeated glyph is translated literally without the window and as the
    // matching sound effect with it. The window holds `嘭 -> "Boom!"` four
    // times and the prompt says to preserve terminology, so the model applies
    // the taught mapping to any name built on a repeated glyph.
    //
    // The gate is the DETECTOR'S OWN LABEL, not a shape test, and that choice
    // is load-bearing twice over. A character-repetition guard in `story.rs`
    // was rejected because 2 of its 3 hits were faithful translations of
    // genuinely repetitive sources -- a shape test cannot tell those apart,
    // but the detector already told us which regions are drawn sound effects
    // (`label == "onomatopoeia"`, the raw label the region promotion keeps
    // precisely because `region_kind` stops answering after promotion). And a
    // skill name like `三重防御` carries `label == "text"`, so its pair stays.
    //
    // The SFX is still translated and still lettered -- this flag only stops
    // its pair TEACHING. `--segment-context` also fixes that page and was measured
    // first: it rewords 78 of 113 live regions (69%) and loses at least three
    // cultivation terms, which is why the cheaper lever exists.
    /// Keep sound-effect pairs out of the story window, so a sound effect's translation
    /// is not reused as terminology. Default: on.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub story_excludes_sfx: Option<bool>,

    // Repeatable. Defaults to `CCWildWords`, then `Arial`.
    //
    // This used to be left empty, because "CCWildWords" is not installed on
    // Windows and asking for it sends the resolver down the bundled-catalog
    // path -- an HTTPS call to huggingface.co. That cost is real but it is
    // once per *process*, not per page: the revision lookup is memoised in
    // `koharu-runtime`'s `REVISIONS` and the .otf itself is cached on disk.
    // Measured on the fixture: the first render goes 0.11s -> 1.68s and every
    // later one is back to 0.11s. A warm server pays it at startup.
    //
    // What we got for it is the whole point of the project: real comic
    // lettering instead of Arial. Pass `--font-family Arial` to go back.
    /// Font family for the lettering; repeat the flag to give a fallback order.
    /// Default: CCWildWords, then Arial.
    #[arg(long = "font-family", value_name = "FAMILY")]
    pub font_families: Vec<String>,

    /// Applies to the whole multipart body, not just the image field.
    #[arg(long, default_value_t = DEFAULT_MAX_UPLOAD_BYTES)]
    pub max_upload_bytes: usize,

    /// How long a request waits for its turn on the GPU before giving up. Once
    /// a translation starts it always runs to completion.
    #[arg(long, default_value_t = 600)]
    pub request_timeout_secs: u64,

    /// Free the models after this many seconds with nothing to do. 0 keeps them
    /// resident for the life of the process.
    #[arg(long, default_value_t = 300)]
    pub idle_unload_secs: u64,

    // Load the models at startup rather than on the first page, moving koharu's
    // per-stage profiling pass off the user's first request. The only flag here
    // that puts anything on the GPU by itself.
    /// Load the models at startup instead of on the first page. The only flag that puts
    /// anything on the GPU by itself.
    #[arg(long)]
    pub warmup: bool,

    // Shut down -- through the same graceful path as `POST /shutdown`, so an
    // in-flight page still finishes -- when the process with this PID exits.
    //
    // `serve.ps1` passes its own PID. Without this, a killed or crashed
    // launcher leaves the server orphaned: Windows children outlive their
    // parents, so it keeps the port, keeps a token nobody can read back, and
    // -- warmed -- holds ~20 GB of VRAM nobody will free. The Hunyuan shim
    // and the HF sidecar carry the same tie; a kill test observed their stack
    // folding in 5.4 s while this process kept answering `/health`, which is
    // the gap this flag closes.
    /// Shut down gracefully, as POST /shutdown does, when the process with this PID
    /// exits.
    #[arg(long, value_name = "PID")]
    pub watch_pid: Option<u32>,

    // Letter translated *dialogue* in capitals, as traditional comic
    // typesetting does. Narration, signs, captions and credits are left alone:
    // they are ordinary prose, and the dialogue role detection already assigns
    // is what tells them apart.
    //
    // Off by default, and it should STAY off now that the default face is
    // CCWildWords. This comment used to say "turn it on once a real lettering
    // font is on the machine"; rendering the fixture in CCWildWords showed
    // that to be wrong.
    //
    // Comicraft faces are drawn caps-first in a specific sense: the uppercase
    // codepoints carry tall capitals and the *lowercase* codepoints carry a
    // second, shorter set of capitals. Mixed-case text therefore already
    // letters in caps, with the height variation the style depends on --
    // visible in the fixture as the tall `B` of "BEAUTIFUL" against the
    // shorter `EAUTIFUL` after it. `to_uppercase` maps every glyph onto the
    // tall form and flattens exactly that rhythm.
    //
    // It remains available for a face without the pairing.
    /// Letter translated dialogue in capitals. Off by default; the default font already
    /// letters in a capitals style.
    #[arg(long)]
    pub uppercase_dialogue: bool,

    // Letter a region whose OCR text is not text: a watermark, punctuation
    // only, one or two stray Latin letters, or the wrong script for the page.
    //
    // **OFF is the default, and a pixel comparison is what set it.**
    // One Chinese test page is a full page of falling petals carrying *no text
    // at all*; detection found 23 regions in the petal shapes, PaddleOCR-VL
    // returned one-to-four character strings for them, and the page rendered
    // with `PINK PIKA`, `HI`, `HMPH`, `HITO` and `20000070` lettered across
    // clean artwork. `untranslated` was empty and `truncated` false, so no
    // number anywhere reported it.
    //
    // The rules are the offline benchmark labeller's, which classified this
    // over saved responses, unable to affect what a reader sees. Validated
    // against 511 labelled segments of a frozen benchmark corpus before the
    // code was written: they refuse
    // **161 of the 161** segments that labeller calls non-text and **0 of the
    // 350** it calls usable. Undeclared source language drops that to 122 and
    // leaves the false-positive count at zero -- the two script rules need a
    // language, and `source_language` on the request is the only place one
    // comes from. See `labels.rs` for why the region cannot supply it -- the
    // region field merely echoes this declaration back.
    //
    // **Free-standing text only.** An erased balloon with nothing put back is a
    // hole, and the first build made four of them on one Japanese test page --
    // including the countdown bubbles reading `3` and `2`. 147 of the 161
    // phantom segments are on artwork, so the restriction costs 9% of the
    // refusals and buys back every empty balloon.
    //
    // **What it cannot do, stated because the number will otherwise look
    // wrong:** the erase mask is built in the detection stage, so the artwork
    // under a phantom box is already inpainted by the time any OCR string
    // exists. This stops the nonsense being drawn; it does not restore the
    // pixels. `--skip-implausible-masks` is the arm that acts on the erase, and
    // it is geometry-only by necessity.
    //
    // Turn this on to A/B the gate against itself rather than argue about it.
    /// Letter free-standing regions whose OCR text does not look like text (a
    /// watermark, punctuation only, stray letters, or the wrong script). Off by
    /// default, so they stay unlettered.
    #[arg(long)]
    pub letter_implausible_text: bool,

    // Letter one utterance twice when the detector found it twice.
    //
    // The pipeline can detect the same text as two regions -- once alone and
    // once inside a larger region that contains it -- and letter both, painting
    // the two translations over each other. Measured against the post-layout
    // collision signal on 100 pages (40 Japanese manga, 60 manhua), **4 of the
    // 10 pairs that really overlap where they were drawn are this**, including
    // the manhua chapter title overwritten by the credits block that repeats it.
    //
    // The gate is ON by default, which no previous collision gate has been, and
    // the reason was that it **cannot lose a word**. That holds in the
    // **strict-containment** branch: there it fires only when one region's
    // translation is a proper substring of the other's and drops the contained
    // side, so every character it removes is still on the page in the region that
    // survives. It hides lettering only -- the erase is untouched, and un-erasing
    // is not wanted here anyway.
    //
    // **It does NOT hold in the identical-string branch, and that is the branch
    // that fires.** Identical strings contain each other, so the tie-break drops
    // one on area and the survivor is somewhere else on the page entirely. Both
    // firings measured on 219 webtoon pages took that branch: one correctly
    // (one drawn `Drill` detected twice) and one wrongly, losing one of two
    // separate 嘿 marks 221 px apart whose ink was already erased. So the
    // ON-by-default justification above is sound for containment and overstated
    // as written; whether that branch should drop at all is an open question,
    // and `--no-duplicate-oriented-overlap` below says the same thing in more
    // detail.
    //
    // Two guards, both measured. Nesting alone over-fires by twenty to one (85
    // nesting pairs on those pages, because `...` is inside every line with an
    // ellipsis), so the frames must also overlap. And the shorter translation
    // must carry at least 3 letters or digits: without that floor the rule
    // drops a region whose whole translation is `!`, because `!` is inside
    // `kuuuuuuh!`. At a floor of 4 it loses `hah!!`. 4 fires, 0 wrong, 0 missed.
    //
    // Turn this on to A/B the gate against itself rather than argue about it.
    /// Letter both copies when the detector finds the same text twice. Off by default,
    /// so the duplicate is dropped.
    #[arg(long)]
    pub letter_duplicate_text: bool,

    // Decide the duplicate gate on the bounding boxes rather than on the frames
    // **as drawn**.
    //
    // **The oriented geometry is ON by default**, so this flag is the arm that
    // turns it off. It first shipped OFF because the evidence was two pages;
    // the corpus measurement below replaced them and turned it on.
    // The flag is inverted at `resolve`, so every reader downstream asks the
    // positive question -- the `no_collision_relief` idiom exactly.
    //
    // The gate asks whether two lettered frames overlap, and `frame()` returns a
    // *polygon* -- a rotated rectangle for angled text. `axis_aligned_bounds`
    // throws that rotation away, and an AABB of a rectangle at 45 degrees is 2x
    // its true area. So two effects that never touch can be reported as
    // overlapping, and because the gate then *drops* one whose ink the inpainter
    // has already erased, the page loses it outright.
    //
    // Measured on a webtoon test corpus (219 pages), where the gate fires
    // exactly twice and the two firings disagree:
    //
    // | page | AABB overlap | oriented overlap | correct |
    // |---|---|---|---|
    // | A -- one drawn `Drill` detected twice, both upright | 194,766 px | **11,128 px** | fire |
    // | B -- two SEPARATE marks reading `Hey`, at 40 and -45 degrees | 572 px | **0 px** | do not fire |
    //
    // A blind 3-seat panel per page was unanimous both ways: on B all three
    // seats found the second mark erased with nothing painted back.
    //
    // **One number carried the default.** Rendering the whole corpus with
    // the oriented geometry changes B by **3,486 px -- identical to what
    // turning the gate fully OFF changes it by** -- while leaving A at
    // **zero**, where turning the gate off costs 13,309 px of garble. So it
    // restores precisely the lost mark and keeps the correct dedup. Everything
    // else is the byte-identical control arm's own floor: 3 pages at 1 px,
    // max|d| = 1. 0 region-count, 0 OCR-source, 0 translation, 0 `font_size`
    // mismatches against the shipping arm.
    //
    // **Two honest limits on that, which the flip does not erase.** The gate
    // fires **twice on 219 pages**, so this is two correct decisions and not a
    // rate; and manhwa (59 pages) fires **zero** times, so nothing here
    // generalises to Korean material.
    //
    // **This is not the whole defect.** Both firings took the *identical-string*
    // tie-break, where the survivor is somewhere else on the page entirely, so
    // `duplicate.rs`'s "cannot lose a word" holds only for the strict-containment
    // branch. Narrowing the geometry fixes the two observed pages; whether that
    // branch should drop at all, or shrink per `request.rs`'s "shrink rather than
    // drop, because a dropped layer cannot be recovered", is an open question.
    /// Make the duplicate check compare axis-aligned bounding boxes instead of the
    /// rotated text frames it uses by default.
    #[arg(long)]
    pub no_duplicate_oriented_overlap: bool,

    // Make the duplicate gate also require the two SOURCE boxes to coincide
    // before a pair may drop -- one ink, read twice, rather than two authored
    // marks whose English happens to collide.
    //
    // **ON by default, on a rendered A/B** (one crop: 220 differing px, both
    // twin heartbeat marks standing). The bare gate decides on the PAINTED
    // frames, and a census of a Japanese test volume showed what that costs:
    // English set into two separate containers routinely overlaps as painted
    // while the source marks never touch, so an authored mark is dropped and
    // its already-erased container ships empty -- twin laugh columns, twin
    // heartbeats, a scream balloon lettered only outside itself, an
    // explanation box emptied into a caption. Every one of those pairs measures source
    // intersection-over-smaller at or under 0.337, while every adjudicated
    // one-ink-read-twice site measures 0.510 or above (16 sites over four
    // corpora, each looked at). `duplicate.rs`'s
    // `DUPLICATE_SOURCE_IOS_FLOOR` (0.42, the hole's midpoint) carries the
    // full table, and its tests pin both edges.
    //
    // Additive, never a replacement: with this ON every drop is a drop the
    // bare gate would also have made, so an A/B's pixel diff is exactly the
    // spared sites. The accepted cost: a pair missing corroborating source
    // geometry letters twice instead of dropping. `--duplicate-shared-source false` is the
    // same-binary control arm.
    /// Drop a duplicate only when the two source boxes overlap as well as the lettered
    /// frames. Default: on.
    #[arg(long = "duplicate-shared-source", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub duplicate_shared_source: Option<bool>,

    // Shrink a text layer that auto-fit would set on top of another one.
    //
    // The one collision rule that can act on what is actually drawn. Detection
    // sees bounding boxes, and box overlap was measured and retired as a
    // predictor -- a fit box is about four times the area of the text set inside
    // it, so two boxes routinely overlap while the ink comes nowhere near. The
    // renderer solves the layout later and knows the real extents, and on 40
    // manga pages the same ink rule fires on 67 box pairs against 7 placed
    // pairs, with a five-judge vision panel finding all 7 genuine and none of
    // the 60 discarded pairs a real collision.
    //
    // **It shrinks; it never drops.** By layout time the inpainter has already
    // erased the artwork under the text, so removing a layer leaves a blank fill
    // where the drawing was -- which is what a panel refused an earlier,
    // dropping version for. Dialogue never yields, and between two free-standing
    // texts the LARGER one does, which is deliberately the opposite of that
    // version's order: dropping asks
    // which loss hurts least, shrinking asks which one has room to give.
    //
    // Bounded: 15% of the size per round, at most 6 rounds, and never below
    // `minimum_font_size`. A collision that survives 0.38x is not a sizing
    // problem.
    // **On by default.** Pass this to turn it OFF and A/B the pass against
    // itself; the flag is inverted at `resolve` so every reader downstream asks
    // the positive question.
    //
    // A five-judge blind panel, 9 matched pairs with 4 controls: **20 of 21 arm
    // preferences favoured it**, all 4 controls were correctly called
    // pixel-identical, and every judge endorsed the shrink itself -- "none of the
    // shrunk effects is too small to read comfortably, and none has been
    // flattened", "at 78.6, 75.0 and 31.2 px they are still the heaviest
    // lettering on their pages". The one dissenting verdict was aimed at the
    // duplicate gate, which shared the arm, and said of this pass: "if this pass
    // shipped on its own I would say ship_it".
    /// Turn off collision relief, which shrinks a text layer that would otherwise be
    /// lettered on top of another one.
    #[arg(long)]
    pub no_collision_relief: bool,

    // Anchor a cut balloon's lettering at the source ink's own center instead
    // of centering it in the visible part.
    //
    // A slice boundary can cut a balloon whose text it does not cut: on a
    // 179-slice Chinese test chapter, all five reported exhibits are
    // balloons the cut crosses while every glyph of the source sits on one
    // side. The balloon's segmentation mask is exactly slice-sized, so a cut
    // balloon's fit frame ends at the canvas edge, and `placement()` centers
    // the block in that clipped frame -- displaced away from the cut by half
    // the amputated span. Measured on that chapter: 23 of 24
    // cut-crossing dialogue balloons letter away from the cut, median 77.8 px,
    // max 445.5 px, while sitting within ~1 px of the clipped frame's center;
    // the reader of the joined page sees text hugging one end of a balloon
    // every other balloon centers in.
    //
    // The anchor is the recognized-from ink's vertical center -- the artist
    // centered that text in the WHOLE balloon, so it is the one position that
    // still reads as centered after the join. Un-clipped balloons already
    // letter a median 6.6 px from it (clipped ones 46.3 px), so anchoring
    // makes a cut balloon behave like a whole one rather than introducing a
    // new policy. The block never leaves the balloon: the anchor is clamped
    // into the fitting range, and a balloon whose fit frame touches no canvas
    // edge is untouched by construction. Displacements under max(16 px, 6% of
    // the fit height) keep today's centering -- the first render turned a
    // clean two-line balloon into a hyphenated three for a 15.8 px shift, so
    // a move a reader could not see is not worth a re-solve.
    //
    // **ON by default, on the adjudicated A/B render**: four of the five
    // exhibits fixed in pixels, re-solved balloons mostly lettering
    // larger, 314 of 335 images byte-identical, zero overflow either arm.
    // This flag is the arm that turns it OFF. Inverted at
    // `resolve`, the `no_collision_relief` idiom, so every reader downstream
    // asks the positive question.
    /// Center the lettering of a balloon cut by a slice edge in its visible part,
    /// instead of at the source text's own center (the default).
    #[arg(long)]
    pub no_edge_anchored_lettering: bool,

    // Turn OFF the free-standing text size cap, so that change can be A/B'd
    // against itself rather than argued about.
    //
    // `lettering::fit_free_text` caps a vertical caption so its translation
    // lays down about as much ink as the Japanese it replaces. It fires on one
    // narrow class of layer -- free-standing, vertical -- and never on
    // in-bubble dialogue, so "the bubble text is small" is not it. Without a
    // switch that is an assertion; with one it is a measurement.
    /// Turn off the size cap for vertical free-standing captions, which keeps their
    /// English close to the ink area of the source text.
    #[arg(long)]
    pub no_fit_free_text: bool,

    // How much of a vertical caption's box the SOURCE ink is assumed to fill.
    //
    // The size cap solves `size = sqrt(f * w * h / (0.42 * chars))`, so this is
    // the term that decides how large the English is allowed to be, and an
    // error in it is an error in the size by its square root.
    //
    // **0.12 is one measurement of one title-page caption** — 29,570 glyph px
    // in a ~250,000 px region — and `lettering.rs` justifies it by *"vertical
    // Japanese is set with generous inter-column air"*. A single-column skill
    // name has no inter-column air at all, so the figure is a category error
    // there and under-sizes the English.
    //
    // **Measured:** this cap, and not the layout box, is what holds one test
    // chapter's skill name at 61.67 px — `fit_free_text` reads
    // `content.source_region()` and never `layer.frame()`, so it is independent
    // of `--turn-unjoined-columns`. Raising it is the only lever that makes a
    // skill name bigger.
    // Left at the measured default; the flag exists so the change can be A/B'd
    // in a render rather than argued about.
    /// Fraction of a vertical caption's box that the source ink is assumed to fill,
    /// used by the caption size cap. Larger values allow larger English.
    #[arg(long = "source-ink-fraction", value_name = "FRACTION",
          default_value_t = crate::lettering::SOURCE_INK_FRACTION)]
    pub source_ink_fraction: f32,

    // Stop translating drawn sound effects, restoring the behaviour that
    // erased them.
    //
    // RF-DETR's `onomatopoeia` class is lettered by default: the region becomes
    // a `TextRegion`, so OCR reads it, the translator receives it as one more
    // segment and the renderer paints English where the kana were. Measured on
    // 40 real pages there are about 2.2 effects per page, so this is not a rare
    // class.
    //
    // The flag exists because that is a visible change to every page, and a
    // default should be measured rather than argued.
    // Turning it off also restores the old mask rule -- an effect is erased
    // only where a bubble contains at least half of it -- because erasing every
    // effect while painting nothing back would be strictly worse than either
    // arm.
    /// Do not translate drawn sound effects; erase one only where a bubble contains it,
    /// as older builds did.
    #[arg(long)]
    pub no_translate_sfx: bool,

    // Sharpen the erase mask with `manga-text-segmentation-2025` before it
    // reaches the inpainter. Off by default, and costs roughly 1.5s per page.
    //
    // The refinement was measured worth **0.19%** of the rendered
    // pixels, which is why it has never had a default or, until now, a switch
    // on this server at all. That verdict was reached on *bubble* pages, and it
    // is sound for them: `fill_uniform_regions` paints the balloon's median
    // colour over the masked pixels before the tile loop, so a glyph-tight mask
    // and a 10px blob produce the same flat fill and precision buys nothing.
    //
    // Lettering sound effects changed the population being measured. Drawn sound effects are
    // now erased wherever they sit, including over open artwork, and that is
    // the one case where mask precision was always supposed to matter -- the
    // blanket `round(max_dim / 1024 * 6)` dilation has no balloon fill to hide
    // behind there.
    //
    // **Measured on that population, and it is WORSE. Do not turn
    // this on for sound effects expecting it to help.** 40 pages, sound-effect
    // lettering on in both arms, only this flag differing:
    //
    // | | off | on |
    // |---|---|---|
    // | detection | 138ms/page | 223ms/page |
    // | layout warnings | 49 | 49 |
    // | pixel change inside effect boxes | - | 7.15 |
    // | pixel change page-wide (control) | - | 0.68 |
    //
    // The segmenter really is acting on effects specifically -- the change is
    // ~10x concentrated inside their boxes -- but it acts in the unhelpful
    // direction, and the reason is structural rather than a tuning miss. The
    // refined mask is INTERSECTED with the dilated box mask
    // (`intersect_masks`), so it can only ever erase **less**. Less erasure is
    // the right instinct when the risk is chewing up surrounding artwork; the
    // defect actually measured is the opposite one, ink left behind that LaMa
    // then reconstructs. Verified in pixels: a bold outlined bell effect comes
    // back as fully legible grey ghost katakana, which is worse than the smears
    // it replaced, and a white-on-black effect keeps more of its ink.
    //
    // `text_mask_padding_iterations` is the one untested knob, since growing it
    // is the only variant that pushes toward *more* erasure. Rate it low: at
    // the padding needed to swallow a bold outlined effect it has recreated the
    // blanket dilation it exists to avoid.
    /// Refine the erase mask with the manga-text-segmentation-2025 model before
    /// inpainting. Off by default; adds about 1.5 s per page.
    #[arg(long)]
    pub refine_text_mask: bool,

    // Denoising steps per tile for `--inpainting rorem-mixed`. RORem's own
    // default is 30, which is what makes it cost 35s a page against LaMa's
    // 0.6s. Exists so that number can be measured instead of assumed.
    /// Denoising steps per tile for --inpainting rorem-mixed. Unset uses RORem's
    /// default of 30.
    #[arg(long, value_name = "STEPS")]
    pub rorem_steps: Option<i32>,

    // How completely `--inpainting flux2-klein` replaces the masked region.
    //
    // **Anything above 0.75 is the same setting**, because the step boundary
    // floors at 4 denoising steps -- 0.8, 0.999 and 1.0 produced byte-identical
    // pages. Only values at or below 0.75 change anything, and they erase
    // *less*. Kept reachable to make that reproducible, not because it is a
    // knob worth turning. See `RoremMixedConfig`'s sibling field for the
    // arithmetic and for the seed-noise figure it corrected.
    /// How completely --inpainting flux2-klein replaces the masked region, from 0 to 1.
    /// Values above 0.75 all behave the same.
    #[arg(long, value_name = "S")]
    pub flux_strength: Option<f64>,

    // Confidence a drawn sound effect needs before it is erased and lettered.
    // Unset keeps the checkpoint's own 0.2 -- the lowest of its four classes,
    // calibrated when this class only ever fed an erase mask. Since effects are
    // now lettered, a false positive costs artwork; see the config doc in
    // `stages/detection.rs`.
    /// Detector confidence a drawn sound effect needs before it is erased and lettered.
    /// Unset keeps the model's default of 0.2.
    #[arg(long, value_name = "T")]
    pub onomatopoeia_threshold: Option<f32>,

    // Grow each region's erase mask to this multiple of its own bounding box
    // instead of by the page-flat ~10px. 1.37 is the optimum reported for LaMa
    // text removal (arXiv 2511.22499). Unset keeps the flat rule.
    /// Grow each region's erase mask to this multiple of its own bounding box, instead
    /// of by a fixed margin of about 10 px. Unset keeps the fixed margin.
    #[arg(long, value_name = "SCALE")]
    pub mask_scale: Option<f32>,

    // Read a text region whose larger side is at least this many pixels with
    // PaddleOCR-VL, whatever `--ocr` selects. Pass `0` to turn it off.
    //
    // **This is the fix for a page of small text coming back as confident
    // nonsense.** manga-ocr and baberu-ocr both resize the crop to a fixed
    // SQUARE (224 for manga-ocr) without preserving aspect ratio, so what the
    // encoder sees is `font_px * 224 / crop_width`. Measured on a synthetic
    // sweep: a 16px font in a 650px box gives 5.5px and manga-ocr recovers 19%
    // of the characters, inventing fluent prose for the rest; PaddleOCR-VL,
    // whose `smart_resize` keeps the aspect ratio, recovers 98%.
    //
    // **Defaults to 448**, twice the 224 input, on a real-page regression check
    // rather than on the synthetic sweep. Over 453 regions of 40 real pages it
    // fires on 3 (0.7%), leaving **37 of 40 pages byte-identical**. All three
    // were opened against the printed page: two were catastrophic manga-ocr
    // failures -- a dense character profile read as invented prose, and a
    // chemistry diagram read as moaning -- and the third was a wash. Zero
    // regressions, so it is on.
    //
    // A crop routed unnecessarily should cost only time, since the 66-region
    // benchmark found manga-ocr and paddleocr-vl-1.6 disagree on kanji in zero
    // regions. Note that claim is about *bubble* crops and has not been tested
    // on large ones: both large crops in the sample turned out to need routing,
    // so there is no measured example yet of a needless trigger.
    //
    // **The one real cost, and it is a first-run cost:** the first region that
    // trips this pulls a ~1.9 GB model if PaddleOCR-VL is not already on disk,
    // and koharu's downloader reports progress only on a channel the GUI reads,
    // so it can look like a hang. The load is announced with `tracing::info!`
    // before it starts. `--large-crop-ocr 0` turns the whole thing off and
    // restores the previous behaviour exactly.
    /// Read a text region whose longer side is at least this many pixels with
    /// PaddleOCR-VL, whatever --ocr selects. 0 turns this off.
    #[arg(long = "large-crop-ocr", value_name = "PX", default_value = "448")]
    pub large_crop_ocr: Option<u32>,

    // The ceiling on `--large-crop-ocr`: a region covering at least this
    // fraction of the page's own area is NOT routed. Pass `0` to turn it off.
    //
    // **`--large-crop-ocr` is a lower bound with no upper bound, and that cost
    // a whole run.** RF-DETR mis-*segments* as well as mis-classifies: one
    // page of the manga test volume produced one `onomatopoeia` box of
    // 1210x1247 on an 844x1200 page -- **1.49x the area of the page it was found on** -- at
    // confidence 0.43, over artwork. It cleared 448, went to PaddleOCR-VL, and
    // returned zero characters. The crop is ~60x a typical text region, so
    // torch's CUDA caching allocator raised its high-water mark ~3.7 GB
    // (28,224 -> 31,965 MiB of 32,607) and never gave it back -- there is no
    // `empty_cache` in koharu's FFI surface. Windows admission control budgets
    // on DXGI `CurrentUsage`, which counts cached-but-free bytes, so the card
    // looked permanently full: `unload_idle` began evicting the 16.5 GiB
    // translation model to admit a 153 MiB detector, and `reservation` then
    // charges an unloaded stage its `peak_bytes`, so it never unlatched.
    // Measured: page median **5,189 ms before, 12,790 ms after**, and 27 cold
    // LLM reloads at a bit-stable 20,521 MiB free.
    //
    // **This is a geometric test, not a content heuristic.** Ink density,
    // detection confidence and text-segmenter fraction were all measured
    // against this same class of mis-segmented box and all three failed -- the
    // segmenter one *inverted*. This asks nothing about the content: a region
    // larger than the page it was detected on is a detector error by
    // construction. A fraction rather than a pixel count, because the two page
    // sizes this project measures on (844x1200 and 1492x1118) would need two
    // different pixel numbers to say the same thing.
    //
    // **Defaults to 0.5**, in a gap with nothing in it: of the 24 routed
    // regions in the run that latched, the largest legitimate one is 0.30x of
    // the page and the outlier is 1.49x. 0.5 is 1.67x above the first, and the
    // largest region the large-crop routing was ever measured on is 0.16x.
    //
    // **What it gives up.** A genuinely huge region with genuinely small glyphs
    // -- a half-page dense prose block -- is the routing's target case, and above the
    // bound it goes back to the engine that fabricates on wide crops. Nothing
    // like it has been seen: the largest real region in ~477 measured is 0.30x.
    // The bound is ON because the harm it prevents is measured and latching
    // while the harm it risks is hypothetical, which is the same standard that
    // turned `--large-crop-ocr` itself on.
    /// Upper bound for --large-crop-ocr: a region covering at least this fraction of
    /// the page is not routed. 0 turns the bound off.
    #[arg(
        long = "large-crop-ocr-max-area",
        value_name = "FRACTION",
        default_value = "0.5"
    )]
    pub large_crop_ocr_max_area: f32,

    // Read a region over `--large-crop-ocr-max-area` with NO engine at all,
    // rather than merely keeping it off PaddleOCR-VL.
    //
    // **On, and that was decided in pixels rather than argued.** This shipped
    // off for one build, on the reasoning that refusing to read a region is the
    // bigger unknown. The volume run settled it: that mis-segmented box is the
    // only region in 2,441 whose handling the area ceiling changes, and reading
    // it with the primary produced `そして、` where PaddleOCR-VL had produced the
    // empty string. That is manga-ocr's documented failure mode -- fluent, not
    // empty -- and the render shows it lettered as a full-panel `AND SO,` across
    // a panel that carries no text whatsoever.
    //
    // **Turning it on cannot rescue artwork**, and it is worth being clear
    // about that: the erase mask is written by the detection stage from the raw
    // detections, before OCR ever runs, so a refused region is still erased
    // either way. All this decides is whether English gets painted on top. The
    // erasure on that page is the separate, deeper defect -- RF-DETR segmenting
    // artwork as `onomatopoeia` at confidence 0.43 -- and the same
    // area-against-the-page test now gates the mask as well, behind
    // `--skip-implausible-masks`, which is **ON** because the
    // pixel comparison came back: 19 of 20 pages byte-identical and that page's
    // whole panel restored. So on the shipping default that page's artwork now
    // survives; the sentence above describes what this flag alone can do.
    //
    // The counter-case is a legitimately huge region: it has huge glyphs for the
    // same reason it has a huge box, so the 224 square would be benign there and
    // dropping it blanks a full-page sign for nothing. Nothing in this volume is
    // such a region -- the ceiling refuses exactly one box, and it is a detector
    // error -- so the measured cost of `true` here is zero and the measured cost
    // of `false` is a fabricated caption over artwork. `--skip-implausible-regions
    // false` is the other arm.
    /// Do not read a region larger than --large-crop-ocr-max-area at all; such a region
    /// is treated as a detector error.
    #[arg(long, value_name = "BOOL", default_value = "true")]
    pub skip_implausible_regions: Option<bool>,

    // Turn a tall vertical free-standing column into its own ink instead of
    // widening it, so the English runs DOWN the column as the artist set it.
    //
    // **ON by default**, after the render was looked at -- as for anything
    // that changes what a page looks like.
    //
    // **The number that made the case, measured on the exemplar with the same
    // binary in both arms**: the turn is not merely bigger
    // but **better contained** -- lettered ink outside the erased area falls from
    // **15,321 px (3.76%) to 13,984 px (3.47%)** while lettered ink rises 7.8%.
    // That is measured against the raw detection bbox plus `INK_MASK_PAD`, never
    // against the JSON, where `fit_width ~ width` is an algebraic identity that
    // would certify a render painting type 273 px per side over clean artwork.
    //
    // `--rotate-free-text-columns false` is the other arm and restores the
    // widened cell exactly.
    //
    // **SCOPED TO SEAM-JOINED PAGES, which is what keeps it off manga.** The
    // gate is `joined_page` -- the caller stating it assembled this image from
    // a run of slices -- and NOT a format classifier: "webtoon or manga?" was
    // measured and rejected as a discriminator, and its counterexample was a
    // JAPANESE webtoon, whose captions a format gate would turn as well. The
    // seam that sets the flag is itself scoped away from manga on purpose, by a
    // tight edge band chosen so it does not fire on a vertical-scroll manga
    // reader. So a manga page is never joined and never turned. The cost, stated
    // plainly: display text wholly inside one slice is not joined either, so
    // this is narrower than "all webtoons".
    //
    // It exists because the BOX, not the size cap, is what holds a skill name
    // small: `--source-ink-fraction` measured 2.10x in the JSON and **zero
    // pixels** in the render. Turning the box swaps
    // the binding constraint from the column's width to its height.
    /// On pages joined from webtoon slices, fit the English of a tall vertical
    /// free-standing column to the column itself, running down it, instead of widening
    /// the box. Default: on.
    #[arg(long = "rotate-free-text-columns", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub rotate_free_text_columns: Option<bool>,

    // Grow a column the detector truncated at a slice edge, and report a
    // sub-floor box against an edge as an `edge_hint`. **ON by default**,
    // after both arms rendered.
    //
    // Both halves feed one consumer, the webtoon seam: it needs a box at each
    // end of a run to plan a join, and on one Chinese test chapter the display
    // column has no usable box on three slices of five -- two score under the
    // floor, and the third's box stops 124 px above its own ink. With this on
    // the join spans four slices and the whole skill name arrives at OCR
    // as one crop; with it off the seam joins the pair it always has.
    //
    // **It changes an UNJOINED page too.** A grown column is past
    // `CROSS_SLICE_HEIGHT_SHARE`, so the cross-slice height gate refuses it on a
    // single-slice request: the artwork survives and the artist's glyphs stay,
    // where otherwise a wrong string is lettered over them. The `resolve`
    // comment on `repair_clipped_columns` carries the evidence.
    /// Extend a text column the detector cut off at a slice edge, and report a small
    /// box touching an edge to the client as an edge hint. Default: on.
    #[arg(long = "repair-clipped-columns", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub repair_clipped_columns: Option<bool>,

    // Read a detected BUBBLE that holds no text region of its own.
    //
    // **ON by default**, on rendered evidence. The detector finds a balloon's
    // shape and its text independently; on one webtoon slice it found the shape
    // at 0.5078 and none of the text, so the balloon reached a reader in
    // Chinese with every counter on the wire correctly reading zero. Every
    // other gate in this project runs downstream of a detection that never
    // happened, which is why nothing else reaches that page.
    //
    // Measured over four populations, 750 pages, 1,565 bubbles: it fires 4 times
    // -- three real untranslated texts, two of them defects nobody had reported,
    // and one spurious bubble on flat skin. **1 false positive in 1,565, 0.064%.**
    // Manga carries 1,330 of those bubbles and produced two hits, both true, so
    // the failure that sank a webtoon-or-manga format classifier does not
    // repeat here.
    //
    // **The first build did not reach the page it exists for, and it damaged
    // the false positive; both arms were rendered.** On the exhibit slice the
    // synthesised region arrived at OCR as **0.6957 of the page**, over the 0.5
    // `--large-crop-ocr-max-area` ceiling, so `--skip-implausible-regions`
    // refused it and the render was **byte-identical** to the off arm. On
    // another webtoon page the spurious bubble was read as `ー` -- the
    // character's own mouth line -- so `--withdraw-unread-masks` never fired (it
    // covers reads that are ABSENT, not reads that are WRONG) and a white dash
    // was lettered across her face. An ordinary page renders byte-identical.
    //
    // That paragraph described the arm before three follow-up fixes.
    // **It shipped.** An ink floor refuses the spurious detection outright, so
    // no region is synthesised for it and the dash never reaches a face; a
    // synthesised region is marked so OCR can refuse to letter a
    // punctuation-only read. The ten pages the rule fires on came back one
    // correct win, one correctly lettered, one refused as a duplicate, seven
    // no-ops, zero defects. The default is pinned by a test below.
    /// Read a detected speech bubble that contains no detected text region. Default:
    /// on.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub read_textless_bubbles: Option<bool>,

    // Lower the `text` admission floor to this score on JOINED pages only,
    // as a REPLACEMENT-ONLY band. **0.20 by default, on the census.** `0`
    // disables, mirroring `--large-crop-ocr 0`.
    //
    // A composite is assembled around a column a slice cut, and squashing it
    // into RF-DETR's fixed square input depresses exactly that column's score:
    // one test composite returns the whole cut column at 0.2383 against the
    // 0.25 floor -- 0.0117 short -- while a top-only fragment at 0.2715
    // survives and reads half the text. The same ink on the base slice scores
    // 0.318. **The census-recommended value is 0.20**: it clears the exhibit
    // by 0.038, and on an unjoined page it touches nothing at all --
    // `joined_page` is the caller stating what it built, never a format guess.
    //
    // **This flag alone is inert BY CONSTRUCTION, not merely in practice.** A
    // band box is never admitted as a region in its own right: it enters NMS
    // as a replacement-only candidate and survives only through the axis
    // tie-break's replace outcome, which does not exist while
    // `--axis-aware-nms` is off. A plain lowered floor would instead have
    // admitted every `[floor, 0.25)` box on every composite as a new region --
    // the exhibit composite's stored log alone carries an unrelated 0.2266
    // strip in that band -- repeating a broad joined-page admission that was
    // measured and rejected. That
    // containment is what makes the floor-only arm the census CONTROL.
    /// Lowest detection score accepted, on joined pages only, for a box that replaces a
    /// conflicting detection. Default: 0.20; 0 disables.
    #[arg(long, value_name = "SCORE")]
    pub joined_page_text_floor: Option<f32>,

    // Prefer the COLUMN-SHAPED box of an NMS-conflicting pair -- same width
    // and much taller, or same height and much narrower -- over the
    // higher-scored one. **ON by default, on the two-corpus census -- and
    // SCOPED to declared Chinese/Korean in `desired_config`**, because the
    // same census refuted it on Japanese manga (10 of 91 pages of a Japanese
    // test volume degraded: stray un-erased glyphs, a name misread). A
    // declared-ja or undeclared request resolves it off whatever this flag
    // says; `false` here turns it off everywhere.
    //
    // RF-DETR returns the same ink twice and containment NMS keeps the higher
    // score, which on both exhibits is the WRONG read: one composite's top-only
    // fragment (0.2715) beats the whole column (0.2383), and another page's wide box
    // (0.373) -- which merges two columns and loses the fourth glyph of a name
    // -- beats the tight name column (0.283). The tie-break fires only when
    // the pair shares one axis within 1.15x and differs on the other by at
    // least 1.30x, and never evicts a box more than 1/0.6 times as confident
    // as the challenger. Everything else keeps today's suppression, op for op.
    //
    // Reaches ordinary slices, not only joined pages -- the second exhibit is
    // an ordinary slice -- which is why the census ran the manga corpus, and the manga
    // corpus is what earned the script scope above.
    /// When two detections of the same text conflict, prefer the column-shaped box over
    /// the higher-scored one. Default: on, for pages declared Chinese or Korean only.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub axis_aware_nms: Option<bool>,

    // Keep the residue of a box the axis tie-break EVICTS, instead of losing
    // it with the box. **ON by default**, on the whole-chapter render.
    //
    // `--axis-aware-nms` above discards the loser whole, and on one test page
    // the loser is the wide box its own doc calls the one that "swallows the
    // gloss": the 73 px it holds left of the winning name column carry the
    // author's red replacement name, 3,898 measured pixels of it. Without this
    // flag they are not erased either -- the erase mask is built from the
    // settled list -- so the page ships English beside untranslated Chinese.
    //
    // With this on, an evicted box's uncovered rectangles are offered back as
    // regions. Two gates stand between offer and admission: a short-side floor,
    // and `ink_within`. The second is the load-bearing one, and in the
    // dangerous direction rather than the obvious one -- a strip that becomes
    // a region also enters the ERASE mask, so admitting an empty one would
    // destroy artwork to letter nothing.
    //
    // Separate from `--axis-aware-nms` so that flag keeps a byte-exact control
    // arm; it cannot act on its own, since a residue exists only where the
    // tie-break evicted something, and that is scoped to declared zh/ko.
    /// Keep the uncovered parts of a box that --axis-aware-nms discards as regions of
    /// their own, when they hold ink. Default: on.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub nms_residue_regions: Option<bool>,

    // Keep a read the pipeline already refuses to letter OUT of the translation
    // request, instead of translating it and throwing the answer away.
    // **ON by default**, on the one-flag render.
    //
    // Measured on a 179-slice Chinese test chapter: 188 of 331 regions are
    // refused and not one of them is ever lettered -- 116 site watermarks, 26
    // hidden by the OCR stage's own veto, 35 illegible reads. The cost is not
    // the wasted 57%; it is that junk sharing a request re-rolls the GOOD text
    // beside it, and *those* pairs are not refused, so they enter the story
    // window and the drift compounds down the chapter.
    //
    // Never empties a page: if nothing would survive the filter, every target
    // is sent exactly as today. `has_work` is therefore identical in both arms,
    // which is what stops an emptied page turning `nothing_translated`'s 502
    // into a silent render of the untranslated source.
    /// Leave text that will not be lettered, such as watermarks and refused reads, out
    /// of the translation request. Default: on.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub skip_unlettered_reads: Option<bool>,

    // Re-draw a strike-through mark the author drew across a name, over the
    // English that replaces it. **ON by default**, on the whole-chapter render.
    //
    // The device is "this name is cancelled, the true one is written beside it".
    // Our render breaks it two different ways on the two pages that carry it:
    // on one the inpainter erases the mark (75% of it), and on the other the
    // mark survives but the English is painted on top, so only 6.6% of what is
    // missing was ever erased. Preserving ink cannot fix the second -- a spared
    // mark is BEHIND the replacement text -- so the mark is measured off the
    // source and drawn again over the finished lettering.
    //
    // Detection is a colour histogram inside the region: the mark is the one
    // coarse colour bucket whose footprint is long, thin, solid and runs most of
    // the region's length. Scoped to a region on purpose -- the same idea was
    // refuted at PAGE scale, where a red art mass dwarfs the mark, and on the
    // first of those pages two such masses do exactly that from outside the box.
    /// Redraw a strike-through line drawn across a name over the English that replaces
    /// it. Default: on.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub strike_through_devices: Option<bool>,

    // Letter free-standing text in the drawn ink's own colour: the fill is the
    // eroded core of what differs from the region's own paper, the weight goes
    // bold when the drawn stroke is heavy, and the halo is recomputed against
    // that fill. **ON by default.** Balloon
    // dialogue is out of scope by construction (the override keys on the same
    // free-standing predicate as the halo); a region whose ink the erosion
    // cannot read letters exactly as the contrast sample always has; and only
    // DARK ink over lighter paper ships -- the one polarity where the
    // paper/ink split cannot have swapped, which is what protects inverted
    // narration and is what a hollow-lettered test render earned. `false`
    // is the byte-exact control arm, and the A/B is meaningful only within one
    // binary: translations resample across builds (fixed seed, drifting
    // requests), so a cross-build pixel compare rewords 61 of 67 pages before
    // styling changes a thing.
    /// Letter free-standing text in the colour and weight of the drawn ink when the ink
    /// is dark on a lighter background. Default: on.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub sampled_ink_lettering: Option<bool>,

    // Read a tall kana-free free-text column a SECOND time with the crop turned
    // 90 CCW, and prefer the turned read. **ON by default**, on the renders.
    //
    // A skill name is a HORIZONTAL line the artist turned sideways, so its glyphs
    // lie at 90 clockwise and the recogniser -- which is never told an
    // orientation -- reads them upright and invents plausible wrong characters.
    // Measured with one paddle call on a whole five-slice joined strip:
    // turned, `苍炎之王·冥霜之王·裂风之王`, byte-exact against the reference;
    // upright, `仓炎岛王王·冥雪岛王王`.
    //
    // **It shipped anyway, because the control render decided rather than the
    // gate** — which is what the paragraph below always said would settle it. The
    // concern it records is still the right one to re-read before widening the
    // rule: a kana census
    // keeps ordinary vertical Japanese out, but upright vertical CHINESE is
    // kana-free as well and would be turned wrongly, and no quality comparison
    // between two well-formed Chinese strings is implementable here -- every
    // engine discards its scores. The control render decides, not the gate.
    /// Read a tall free-standing column that contains no kana a second time, turned 90
    /// degrees, and prefer that read. Default: on.
    #[arg(long = "reread-rotated-columns", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub reread_rotated_columns: Option<bool>,

    // Sweep a free-standing region whose read came back EMPTY through the OCR
    // sidecar's rotation grid (12 coarse + 4 refine angles, rotated in the
    // sidecar with the exact PIL call the measurements used) and let
    // `upright_select` pick a read or decline. **ON by default**, after the
    // chapter render was looked at: 14 recoveries, 8 pages changed, six
    // display titles lettered BESIDE their preserved originals. The known
    // price: two cells lettered junk ("REAL ESTATE"/"DANCE") because their
    // true glyph is Pareto-unreachable from the decoder score; the ship floor
    // below removes them.
    //
    // The population is the refused boxes: display titles and drawn effects
    // rotated 47-130 degrees, outside the pipeline's representable ±45.
    // Measured on a 25-crop test population: **13/17 text reads recovered**
    // against a native floor of 1/17 and an angle-oracle ceiling of 14/17, with
    // **0/8 artwork fabrications surviving the downstream gates** -- the
    // spotting gate, `Refusal::ImageDescription` and the script rules all stay
    // in the recovered read's path, and the measured spot coverage of the wired
    // pages passes 16/17 text boxes while failing 8/8 artwork boxes.
    //
    // **The pass also carries a ship floor**: a pick under three RAW glyphs
    // declines to silence. Offline that trades the fit score to 11/17 (the `快`
    // and `轰` single-glyph EXACTs); on the rendered chapter it costs zero
    // lettered recoveries and removes every junk lettering the pass ever
    // produced ("REAL ESTATE", "DANCE" twice, a stray "1" -- all picks of at
    // most two glyphs, on boxes whose true glyph is unreachable from the
    // decoder score).
    //
    // Cost when on: sixteen sidecar reads per EMPTY free-standing region --
    // nothing else pays anything. Off, the pass does not exist and the binary
    // behaves byte-identically to before the flag existed.
    /// Re-read an empty free-standing region at several rotation angles through the OCR
    /// sidecar, and keep a read only when one is chosen. Default: on.
    #[arg(long = "upright-pass", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub upright_pass: Option<bool>,

    // Whether a SYNTHESISED bubble read (`--read-textless-bubbles`) is read a
    // SECOND time with the crop turned 180 degrees, keeping the flipped read
    // only when its confidence beats the upright read's by the flip margin
    // (`stages::ocr::FLIP_CONFIDENCE_MARGIN` -- `0.17`, a borrowed and
    // deliberately unfitted starting value; the flip's own render is the
    // authority on it).
    //
    // **ON by default, on its render.** The A/B changed exactly ONE page of
    // 335 images (2.57% of its pixels), the balloon body survived intact,
    // every control was byte-identical, and the chapter total sat inside noise
    // (306 s ON vs 322 s OFF). The one firing: upright junk at 0.4978 lost to
    // the flipped `冥霜之用..` at 0.6847 -- delta 0.187 against the 0.17
    // margin, a close call the flip's log line now records for any future
    // re-fit. The population is the ~180-degree rotation class: one slice's
    // upside-down `冥霜之王！` balloon,
    // synthesised by `--read-textless-bubbles`, read upright as fluent junk
    // (three different hallucinations across stored arms) and lettered as
    // confident nonsense.
    // No other lever reaches it: `mask_angle` represents ±45 degrees only and
    // the upright pass sweeps refused FREE-STANDING targets, while a
    // synthesised region is dialogue by construction. Scored engines only --
    // `manga-ocr`, `baberu-ocr` and Ollama report no confidence, never flip,
    // and never pay the second read. Cost when on: ONE extra read per
    // synthesised region, and the 2,519-bubble census holds twelve textless
    // bubbles, so a chapter typically pays well under a second.
    /// Re-read a bubble found by --read-textless-bubbles turned upside down, and keep
    /// that read when it is clearly more confident. Default: on.
    #[arg(long = "flip-reread-bubbles", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub flip_reread_bubbles: Option<bool>,

    // Whether a sparse or decline-carrying page may buy ONE HunyuanOCR
    // spotting call and MINT lettered regions for display runs the detector
    // never boxed. **ON by default, on the rendered chapter checked against
    // the official edition** (zero 500s, three rescued display texts matching
    // the official lettering; 328/335 images byte-identical to the OFF arm;
    // chapter cost 300 s -> 775 s, the spot call on ~every sparse page).
    //
    // The population: a diagonal display banner with no covering box on the
    // host's own bytes, another page's two display runs, plus the census's six
    // missed-display pages, all sparse. Admission is GEOMETRIC first -- area
    // floor 2.5% of page, ceiling 0.5, an overlap fence against every existing
    // box -- because the watermark plate was measured MANGLED on 14 pages, so
    // a string rule can only be a belt. A surviving box is swept at 16 angles,
    // `upright_select` decides, and a shipped read still passes
    // `withdraw_from_mask` before anything is minted. Sidecar route only; the
    // ja-paged arm structurally never reaches it. Cost: one spot call
    // (~0.7 s) on trigger pages, ~3.7 s of sweep per surviving box, single
    // digits per chapter (the measured yield ceiling).
    /// On sparse pages, make one extra HunyuanOCR text-spotting call to find display
    /// text the detector missed, and letter it. Default: on.
    #[arg(long = "spot-rescue", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub spot_rescue: Option<bool>,

    // Whether a shipped spot rescue also joins the ERASE mask. **ON by
    // default, on its OWN render** (against the lettering arm: exactly two
    // images change of 335, both looked at and better -- one band
    // reconstructs clean under its lettering, one drawn glyph removed -- no
    // big-box erase damage, 780 s vs 775 s). It is a separate flag because a
    // rescued display box is far over `rescue_narrow`'s 256 px bound, and a
    // big-box erase is costly when it is wrong; the render is what earned the
    // default.
    /// Also erase the source text that --spot-rescue finds before lettering over it.
    /// Default: on.
    #[arg(long = "spot-rescue-erase", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub spot_rescue_erase: Option<bool>,

    // Whether a WIDE scream-read spot mint becomes the
    // mark-replacement device: the erase scoped to the drawn mark's own
    // enclosed ink instead of the rotated box's hull (which destroyed one test
    // burst's lower lobe -- 247,342 px of art rebuilt as streaks),
    // and ONE styled replacement lettered along the ink's principal axis --
    // gradient fill from the mark's own bright end to its dark end, halo
    // from its own bright ring, heavy weight -- the treatment the licensed
    // page demonstrates. Predicate: spot mint AND `!rescue_narrow` (256 px
    // short side, the bound that separates the exhibit from two small genuine
    // display marks) AND a scream-shaped read (one
    // glyph repeated into at least half of a 3+-glyph run). Inert unless
    // `--spot-rescue-erase` is also on. **ON by default, on the rendered
    // A/Bs** (a Korean test chapter: exactly one fire in 67 slices -- the
    // exhibit -- and 143/144 PNGs byte-identical, composites included; a
    // Chinese control chapter: zero fires, 340/340 byte-identical, 0
    // differing pixels).
    // `--replace-scream-marks false` is the same-binary control arm.
    /// Replace a large drawn scream mark found by --spot-rescue with one styled English
    /// mark along the drawn mark's axis, erasing only the mark's own ink. Needs
    /// --spot-rescue-erase. Default: on.
    #[arg(long = "replace-scream-marks", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub replace_scream_marks: Option<bool>,

    // Whether a JOINED page (a seam composite) may buy the spot call too,
    // regardless of how many regions its detector proposed. **ON by default,
    // on the rendered chapter** — with the paint-back part gate in place,
    // turning this on changed exactly 2 of 179 reader-final slices and
    // delivered the joined `凌霄境界・三重防御` column ("SKYREACH REALM." /
    // "TRIPLE DEFENSE!"), 177/179 byte-identical as the A/A, chapter cost
    // inside run variance (292.1 s vs 295.7 s). `--spot-rescue-joined false`
    // is the control arm.
    //
    // The population: a display run CUT by a slice boundary -- the
    // `凌霄境界・三重防御` column, cut through `三` between two slices,
    // undetected on both slices AND on their joined composite. Probed at debug
    // level: HunyuanOCR's spotting boxes the whole column (`[112, 14, 235,
    // 998]` norm-1000) -- on the base slice the overlap fence correctly drops
    // it against the partial `凌霄境界` region, and on the COMPOSITE nothing
    // overlaps it, so the ordinary admission predicate admits it; the only
    // missing piece was the trigger, because a composite carries its
    // neighbours' ordinary regions (3 on that composite, one over the sparse
    // threshold of 2). A composite exists BECAUSE something crosses the cut,
    // so it is exactly the rescue's population. Cost: one spot call per
    // composite, ~26 on a 179-slice chapter.
    /// Allow the --spot-rescue call on pages joined across a slice boundary too,
    /// however many regions they have. Default: on.
    #[arg(long = "spot-rescue-joined", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub spot_rescue_joined: Option<bool>,

    // How much MORE confident the UPRIGHT read must be before it beats the turned
    // one. **Ships at `0.17`**, on the rendered evidence below. Pass `inf` to
    // turn the ranking off -- no finite confidence gap clears it, so the turned
    // read wins unconditionally, which is the behaviour before the ranking
    // existed. **`0.0` is NOT the off value**: a zero margin makes a tie enough
    // for the upright to win, the opposite of off (pinned in
    // `choose_orientation`'s tests).
    //
    // The turned read used to win whenever it was non-empty, because nothing could
    // rank two reads of one crop. Paddle's decoder now surfaces its own
    // length-normalised sequence probability, and so does the shipping hunyuan
    // sidecar (`inference_scored`,
    // `exp(mean_logprob)` -- the like-for-like quantity); `manga-ocr` and
    // `baberu-ocr` still return `None` and the ranking is skipped for them
    // rather than scoring them zero.
    //
    // **Asymmetric deliberately.** The turn exists to rescue sideways columns, so a
    // tie leaves it alone; only an upright read that is clearly more confident wins.
    //
    // Measured twice on all 179 slices of a Chinese test chapter, once per
    // engine. Paddle: the turn wins its two GOOD pairs by 0.0398/0.1078, the
    // upright its two by 0.2380/0.5449 -- any value in `(0.11, 0.23)` right or
    // neutral on all five differing pairs. Hunyuan: thirteen turn pairs, every
    // adjudicated displacement at delta >= +0.55, every correct turn at delta
    // <= -0.54, washes immune -- any margin in `(0.03, 0.44)` separates them,
    // and the paddle-fitted band sits wholly inside it. `0.17` is the value the
    // chapter was RENDERED at: exactly 3 of 179 pages change, all three fixes
    // ("SKYREACH REALM."/"BOOM!" on one, a skill name on another, "Wind King"
    // on the third), and the rescue population -- the sideways columns the
    // turn exists for -- is byte-identical. All three pages looked at.
    /// How much more confident an upright read must be to beat a turned read of the
    /// same column. Default: 0.17; inf always keeps the turned read.
    #[arg(long = "orientation-confidence-margin", value_name = "MARGIN")]
    pub orientation_confidence_margin: Option<f64>,

    // Grow an ONOMATOPOEIA crop by this many pixels and read it a SECOND time, so
    // the two reads can be compared. Unset disables it, which is the default and
    // is byte-identical to the behaviour before it existed.
    //
    // **Reported, never acted on.** The grown read is logged beside the upright
    // one and then dropped. It does not reach the region's text, the mask, the
    // eraser or the translator, and nothing is gated on it.
    //
    // **The question it exists to answer:** a real drawn glyph should survive a
    // few pixels of extra background; a shape matched off artwork should not.
    // Re-running the IDENTICAL crop tells nothing, so a tell must vary the
    // GEOMETRY. It is scored against corrected 9 ARTWORK / 9 DRAWN-GLYPH
    // labels on a Chinese test chapter, where two boxes first labelled
    // fabrications are REAL sound effects the published edition letters as
    // "BOOM!".
    //
    // **Grow only, never shrink**, and a box the clamp refuses to grow is dropped
    // rather than scored: an unperturbed pair always agrees, and reading that as
    // "stable" is a false negative pointing the wrong way.
    // `stages::ocr::crop_grown` holds the detail.
    /// Diagnostic: re-read each sound-effect crop grown by this many pixels and log
    /// both reads. Unset by default; the second read is never used.
    #[arg(long = "perturb-reread-grow-px", value_name = "PX")]
    pub perturb_reread_grow_px: Option<NonZeroU32>,

    // Offer the free-text column TURN to a column on an UNJOINED page as well.
    // **ON by default**, on the rendered cost. Pass `false` for the other arm.
    //
    // On a joined page the turn already hands back the ink's own bbox — the
    // exemplar measures `fit_width / width` **1.00**, against the 2.92x this
    // flag was written for. An unjoined column still takes the widened cell, whose
    // width is driven by HEIGHT: `max(own_width, own_height * 0.6)` hands a
    // 145x626 column **375.6 px**. Both erase-mask builders key off the RAW
    // detection, so the English is lettered across artwork nothing cleaned.
    //
    // **The predicate claims ordinary upright vertical Japanese too** — 366
    // height-driven free-text regions on a measured baseline, 38.3% of
    // manga-ja free-text — and detection is never told the source language, so it
    // cannot tell those from a sideways display column. That was the argument for
    // holding it off; the render below overtook it.
    //
    // **The render settled it and the cost is bounded to EFFECTS.** Three
    // display columns 2.29x / 2.12x / 1.40x → **1.00**, against one page's
    // `WH-`/`WIN` and another's `*TAP*` lettered sideways at 0.169% and 0.046%
    // of their pages. Dialogue never reaches here — balloon text takes the bubble
    // path — so no balloon changes at any setting.
    /// Apply the column fit of --rotate-free-text-columns on pages that were not joined
    /// from slices as well. Default: on.
    #[arg(long = "turn-unjoined-columns", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub turn_unjoined_columns: Option<bool>,

    // Scope a watermark verdict to the site's OWN TEXT, instead of refusing every
    // character the detector merged with it. **ON by default**, on the renders.
    //
    // A site can stamp its plate horizontally coincident with a display column,
    // so a correctly-drawn box contains both and one watermark condemns the
    // skill name beside it. Nothing merged anything; there is no upstream split
    // to make.
    //
    // **The render is what decided.** The cost is real and bounded: a
    // surviving region is erased and inpainted whole, plate included. On the
    // measured chapter only one page moved, from refused to a correctly read
    // character name, and that is the one genuine loss of story text in it.
    /// When a region mixes a site watermark with other text, refuse only the
    /// watermark's text instead of the whole region. Default: on.
    #[arg(long = "scope-watermark-refusals", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub scope_watermark_refusals: Option<bool>,

    // Let a script or punctuation refusal reach a DIALOGUE-role region on a
    // positively declared zh/ko page, and take the same read out of the erase
    // mask -- a paired lever, both halves on one flag.
    //
    // The free-standing gate exists because a refused in-bubble read leaves an
    // erased balloon with nothing in it. The pairing is what makes the
    // exemption sound: the mask half (`withdraw_from_mask`'s script arm, and
    // `illegible_text`'s `letters == 0` arm for punctuation) means the drawn
    // ink was never erased, so refusing the lettering leaves the balloon AS
    // DRAWN. Measured population, two Korean test chapters: a drawn groan
    // misread as a single Han character at 0.317, erased, and lettered as a
    // pronoun; a stylized red sound effect erased for a wrong translation; a
    // 16x16 detection on a character's PUPIL read `·`, the iris erased and the
    // dot re-lettered.
    //
    // **ON by default**, on a read of the censused chapters (zh control
    // byte-identical 340/340; every exhibit looked at off-vs-ON, zero blank
    // holes). `false` is the census off arm.
    /// On pages declared Chinese or Korean, leave a dialogue region whose read is in
    /// the wrong script, or is punctuation only, as drawn: neither erased nor lettered.
    /// Default: on.
    #[arg(long = "leave-misread-bubbles", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub leave_misread_bubbles: Option<bool>,

    // On a positively declared KOREAN page, treat a read of three or more
    // scripted letters carrying NO hangul as a script mismatch -- refusal and
    // mask withdrawal together.
    //
    // The existing ratio arm divides foreign counts by `scripted()`, which
    // includes latin, so an all-Latin misread of drawn hangul (a short Latin
    // word or acronym) can never reach `FOREIGN_SHARE`, and two Latin letters
    // plus one Han glyph score 1/3 against the 0.34 bar. Korean is written in
    // hangul. Authentic English display art is refused-AS-DRAWN under this
    // rule -- for a Latin target language the art already says what the
    // lettering would.
    //
    // **ON by default**, with its pair above, on the same read.
    /// On pages declared Korean, treat a read of three or more letters with no Hangul
    /// as a script mismatch. Default: on.
    #[arg(long = "korean-script-strict", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub korean_script_strict: Option<bool>,

    // On a positively declared KOREAN page, buy ONE re-read from the reserve
    // engine (PaddleOCR-VL, +24 px padded crop) for a DIALOGUE-role read the
    // misread lever refuses, admitting it iff the re-read is hangul-majority
    // -- script membership, never a confidence comparison.
    //
    // The lever keeps the drawn ink instead of lettering a wrong word; this
    // recovers the READ so the balloon letters English instead of shipping
    // raw. Measured on the censused 19-region refusal population: the reserve
    // on padded crops reads 4 of the 6 genuine dialogue targets EXACTLY where
    // hunyuan re-reads recover none. SFX/onomatopoeia and punctuation-only
    // refusals are excluded -- the official release ships those as drawn.
    // Costs one paddle load (~1.9 GB, first firing only) plus a handful of
    // reads per chapter.
    //
    // **ON by default**, on the rendered evidence: exactly one page differs
    // across both censused chapters (a recovered groan lettered in English,
    // balloon intact, looked at; 143/144 and 97/97 PNGs byte-identical), zero
    // re-admissions. This wires the reserve engine into the shipping path --
    // the reserve doing reserve work. `false` is the census off arm.
    /// On pages declared Korean, re-read a refused dialogue region once with
    /// PaddleOCR-VL and accept the read if it is mostly Hangul. Default: on.
    #[arg(long = "reread-refused-dialogue", value_name = "BOOL",
          num_args = 0..=1, default_missing_value = "true")]
    pub reread_refused_dialogue: Option<bool>,

    // Also keep a region over `--large-crop-ocr-max-area` out of the **erase
    // mask**, so the artwork under it survives.
    //
    // **This is the flag that can actually save that mis-segmented page, and
    // `--skip-implausible-regions` cannot.** That one decides whether the box is
    // *read*; the erase mask is written by the detection stage from the raw
    // detections before OCR has run at all, so the 1210x1247 `onomatopoeia` on
    // an 844x1200 page -- 1.49x the area of the page it was found on -- is
    // erased and inpainted in every arm measured without this flag.
    // Refusing to read it only stopped `AND SO,` being lettered on top of the
    // damage.
    //
    // **A fourth discriminator for the mis-segmented-artwork class, and the
    // first one that is not a content heuristic.** Ink density, detection
    // confidence and text-segmenter fraction were all measured against this
    // class and all three failed -- the segmenter one *inverted*, finding 31%
    // text in the box that had to go and 0% in the box that had to stay. This
    // asks nothing about the content: a region larger than the page it was
    // detected on is a detector error by construction. The same geometry, the
    // same `--large-crop-ocr-max-area` fraction, pointed at the mask instead of
    // at the reader.
    //
    // **ON, and a pixel comparison is what turned it on.**
    // `clean_only`, 20 consecutive pages of the volume, both arms on one binary:
    // **19 of 20 pages byte-identical and the mis-segmented page changing 17.6%
    // of its pixels.** With the gate off LaMa erases a character's whole figure
    // to blank white smears; with it on the page is indistinguishable from the
    // source. Collateral over the 20 pages falls 2.62% -> 1.74%, and residue is
    // unmoved (20.2 -> 20.7, and that +0.5 is that page's own un-erased ink
    // being counted as residue, which is what "do not erase this" means).
    //
    // **The residual risk, stated because it did not appear rather than because
    // it cannot:** a legitimately enormous region -- a full-page sign, a
    // splash-page effect -- would stop being erased and would render with the
    // Japanese still under the English, which looks worse than a missing
    // translation. Over all 2,441 regions of this volume the largest legitimate
    // box is **0.298x** of its page against the phantom's **1.490x**, a 5.0x gap
    // with the 0.5 ceiling between them and nothing else near it. That is one
    // volume of one title. `--skip-implausible-masks false` is the other arm.
    //
    // Only `text-mask` is gated. `bubble-mask` is a layout polygon the editor
    // reads and no inpainter touches.
    /// Do not erase a region larger than --large-crop-ocr-max-area, so the artwork
    /// under it survives.
    #[arg(long, value_name = "BOOL", default_value = "true")]
    pub skip_implausible_masks: Option<bool>,

    // Take a region **OCR could not read** back out of the erase mask, so the
    // artwork under a detector false positive survives.
    //
    // The sibling of `--skip-implausible-masks`, and the stronger of the two:
    // that one refuses a box for being implausibly large, which is a proxy for
    // "not really text". This one refuses it for containing nothing that could
    // be text in any script, which is the thing itself, and it can only be asked
    // after OCR has run.
    //
    // **Why it exists.** `labels::hide_implausible` already refuses these
    // regions correctly -- but it runs here, after `Pipeline::execute` returns,
    // while the erase mask is written during the detection stage before OCR has
    // read a character. So the refusal was always right and always too late.
    // Measured on three Chinese webtoon slices of chapter 381206: **15 regions,
    // 0 lettered, 15 erased**. Five birds against a gradient sky were labelled
    // `onomatopoeia`, read as `↓ Y V √ 1`, refused for lettering, and erased out
    // of the sky with nothing drawn in their place -- 0.901% of that page
    // destroyed for zero text.
    //
    // **On by default**, unlike its sibling, because the render a
    // pixel-changing default needs exists for it. `false` is the
    // other arm and restores the previous behaviour exactly.
    //
    // Inert under `clean_only`, which runs no OCR stage and therefore has no
    // verdict to act on -- so do not measure this arm with `--clean-only`.
    /// Do not erase a region whose OCR read holds no text in any script, so the artwork
    /// under a false detection survives.
    #[arg(long, value_name = "BOOL", default_value = "true")]
    pub withdraw_illegible_masks: Option<bool>,

    // Take a region refused for SIZE back out of the erase mask, not just away
    // from OCR.
    //
    // **It breaks an interlock between two flags that both ship ON.**
    // `--skip-implausible-regions` drops the box before it can become an OCR
    // result, and `--withdraw-illegible-masks` only walks results — so the
    // reader's refusal deletes the very verdict the withdrawal needs, and the box
    // is erased with nothing painted back. Over a benchmark run,
    // 22 such boxes on 457 pages of manhua, manhwa and webtoon: **18 erased.**
    //
    // **The obvious alternative was measured and is worse.** Turning
    // `--skip-implausible-regions` off rescues only **8 of 18** — exactly the
    // subset whose read happens to fire `withdraw_from_mask` — leaves 10 erased
    // and marginally worse, and newly breaks a page that was pixel-perfect,
    // painting `SONG` over intact artwork to render one character. That the
    // rescued 8 are precisely the withdrawal's own population is the argument for
    // this flag: the withdrawal is the right mechanism and the OCR read is not
    // needed to justify it.
    //
    // **ON by default, on a render and a unanimous panel.** The render:
    // **18 of 18 erasures fixed, 0 broken, nothing lettered**, with collateral
    // confined to five pages at max|d| = 1 and 0 pixels over threshold, which is
    // the byte-identical control arm's own floor.
    //
    // **A blind panel then agreed 15 of 15, including a seat instructed to argue
    // against it**, which reported that the pixels did not support the case in any
    // of the five live comparisons. Their shared reasoning is the one that decides
    // this flag: **the erase buys nothing.** Neither arm letters anything into
    // these boxes, so a reader's comprehension is identical either way — which
    // reduces the question to *the original drawing* against *whatever LaMa
    // invents*, and the fill is a visible defect in 4 of 5. On one page it turns
    // the right half into a white vertical smear; on another the erase spills past
    // the panel border and strands smudges in the blank gutter.
    //
    // **What it costs, stated plainly:** an untranslated source sound effect stays
    // visible. That is what the pipeline already does with these boxes — it has
    // never translated them — so the flag changes whether the drawing survives, not
    // whether the reader gets a translation.
    //
    // `false` is the other arm. Independent of `--withdraw-illegible-masks`:
    // turning that off must not silently turn this off too, since the coupling is
    // the defect this exists to remove.
    //
    // **Honest limits, which the default does not erase:** the panel judged 5 cases of
    // 22, and manga-ja was never rendered in that run — it carries 1 such region
    // and that one is already spared by the raw-bbox gate.
    /// Do not erase a region that was refused for its size before OCR.
    #[arg(long, value_name = "BOOL", default_value = "true")]
    pub withdraw_unread_masks: Option<bool>,

    // Show the inpainter a glyph cut by the tile grid as ONE shape, instead of
    // two half-shapes with each other's un-erased ink as context.
    //
    // **The residue this removes is LaMa's own output, not un-erased ink**, and
    // that took a measurement to establish: the erase mask already covers the
    // effect (0 changed pixels outside the detection boxes on 381206/133), and
    // 95% of the surviving stroke was masked and then repainted.
    //
    // The tiler walks a fixed 512 grid and each tile's crop reaches 128px past
    // its core, so when a blob crosses a grid line the neighbouring half arrives
    // as unmasked context touching the hole. LaMa continues the stroke into it,
    // and the next tile reads that back out and continues it again. Residue by
    // distance to the nearest grid line on 133: **42.5% at 0-32px, 45.9% at
    // 32-64, 15.5% at 64-128, 1.4% beyond** -- a cliff exactly at the context
    // radius.
    //
    // **It cannot erase artwork.** The flood only marks pixels already in the
    // page mask, and the compositor writes only inside the tile core where that
    // mask is set, so the page's erased footprint is bit-identical either way.
    // That property is why this is safe to ship beside
    // `--withdraw-illegible-masks`, which exists to stop artwork being erased.
    /// Inpaint a mark that the inpainter's tile grid cuts as one shape instead of two
    /// halves, which avoids leftover streaks.
    #[arg(long, value_name = "BOOL", default_value = "true")]
    pub seam_safe_erase: Option<bool>,

    // Write the settled per-region detection masks and the assembled inpaint
    // masks as PNGs under this directory. A debug instrument for the
    // mark-replacement design -- the masks are invisible from the wire
    // (`KoharuLayoutDetection.mask` is skip_serializing), and four rendered
    // attempts at that design failed on exactly that blindness. Absent -- the
    // default, and the only shipped state -- writes nothing and is
    // byte-identical to before the flag existed. A dump failure warns in the
    // log and never fails the request.
    /// Debug: write the detection and inpainting masks as PNG files under this
    /// directory. Unset by default.
    #[arg(long, value_name = "DIR")]
    pub debug_mask_dir: Option<String>,

    // JSON of pinned English for drawn sound effects. Repeatable; **earlier
    // files win**, so pass the curated list before an imported one.
    //
    // Measured over 40 pages: effects do not drift between pages, so this is
    // not a consistency fix. It exists because 22% of genuine effects came
    // back as romaji (`きゃっ` -> `Kya!`). Off unless given, because a large
    // imported set would otherwise restyle effects the model already letters
    // well, and that has not been measured.
    /// JSON file of fixed English for drawn sound effects. Repeatable; earlier files
    /// win. None by default.
    #[arg(long = "sfx-dictionary", value_name = "FILE")]
    pub sfx_dictionaries: Vec<std::path::PathBuf>,

    // Add the ink actually found inside each detection box to the erase mask,
    // via Otsu plus connected components. No model, no download.
    //
    // The counterpart to --refine-text-mask, which measured negative because it
    // INTERSECTS and can only erase less while the defect is ink left behind.
    // This one is UNIONed and can only erase more.
    /// Add the ink found inside each detection box (Otsu threshold plus connected
    /// components) to the erase mask. Off by default.
    #[arg(long)]
    pub ink_mask: bool,

    // Hand libtorch's cached VRAM back to the driver whenever admission control
    // evicts a model. On by default; pass `false` for the other arm.
    //
    // **This is the root fix for the eviction latch, and it is the only one
    // that is about the mechanism rather than about one trigger.** Torch's
    // caching allocator never returns a segment on its own, and before this
    // there was no `emptyCache` anywhere in the koharu tree to ask it with. So
    // the single largest input the vision stages ever see sets a floor for the
    // whole process: one 1210x1247 region mis-segmented out of an 844x1200 page
    // raised the allocator's high-water mark by ~3.7 GiB (28,224 -> 31,965 MiB
    // of a 32,607 MiB card), and a 1280x1214 webtoon seam crop did the same
    // thing on a different chapter.
    //
    // Those bytes are free as far as torch is concerned and *used* as far as
    // the driver is concerned, and koharu budgets on the driver's figure --
    // DXGI `QueryVideoMemoryInfo`'s `CurrentUsage` on Windows. So
    // `admission_plan` starts evicting; the sweep unloads **every** resident
    // model rather than just enough of them; and re-admitting an unloaded stage
    // is then charged its whole `peak_bytes` instead of its incremental
    // `workspace_bytes`, roughly an 8x jump, which is more than was ever free.
    // The branch never flips back. Measured cost of that latch on a real
    // volume: 5,189 ms per page before it, 12,790 ms after, with the 16.5 GiB
    // translator reloaded 27 times.
    //
    // **What it is not.** `emptyCache` releases a segment only when no live
    // block remains inside it, so a cache fragmented by one surviving
    // allocation per segment can return far less than the ~9.4 GiB gap the log
    // implies. It also synchronizes the device on the way, which costs
    // milliseconds inside a path that is already the expensive one. Whether the
    // trade pays is the measurement, not the mechanism -- which is why there is
    // a flag here at all rather than an unconditional call.
    //
    // Ignored on a CPU or Vulkan device: nothing is registered to empty.
    /// Return libtorch's cached VRAM to the driver whenever a model is unloaded to make
    /// room. Ignored on the CPU.
    #[arg(long, value_name = "BOOL", default_value = "true")]
    pub release_cached_vram: Option<bool>,

    // Ask a stage whether the page has anything for it before paging its
    // weights in, and skip it outright when the answer is no. On by default;
    // pass `false` for the other arm.
    //
    // **The load was always unconditional and the check always came after it.**
    // `stage_runner` calls `Stages::load` and only then `process`, while
    // `Translator::translate` early-returns on an empty segment list -- so a
    // page with no text pays a full cold load of the 16.5 GiB local LLM to be
    // handed nothing. That is free whenever the weights happen to be resident,
    // and costs a whole cold start every time residency has evicted them, which
    // under memory pressure is every page.
    //
    // **What it is worth is a property of the page set, not of the code, and
    // the two page sets this project measures on disagree sharply.** Counted
    // from the run directories: 80 of 219 slices of webtoon chapter 381206
    // (36.5%) come back with zero regions, 66 of those paid the load, and those
    // 66 cost **686 s of a 2,351 s run** -- 29%. The same count over the 213
    // pages of the manga volume is **8 (3.8%)**, every one of which found the
    // weights already resident and cost 71 ms, for 0.6 s of 1,386 s. Whole
    // manga pages nearly always carry dialogue; a strip sliced into fixed-height
    // images is a third empty. Expect a webtoon-shaped win and a manga-shaped
    // nothing.
    //
    // **It is independent of the eviction latch**, and pays either way. The
    // latch decides *how often* the weights are gone when an empty page arrives;
    // this decides whether an empty page reloads them at all. Fix the latch and
    // the 66 becomes a smaller number, not zero -- admission control still
    // evicts legitimately.
    //
    // Only translation can answer the question today. Detection, OCR and
    // inpainting take the default `true` and behave exactly as before.
    /// Skip a stage, without loading its model, when the page has nothing for it to do.
    /// Only translation uses this today.
    #[arg(long, value_name = "BOOL", default_value = "true")]
    pub skip_empty_stages: Option<bool>,

    // How much VRAM one cold page is assumed to need, for the pre-flight that
    // answers 507 instead of letting a full card abort the process.
    //
    // Defaults to 4 GiB under `--provider ollama`, and to 21 GiB under
    // `--provider local`, which is also charged for koharu's own translation
    // model -- 13.26 GB of `gemma4-26b-a4b-it` weights plus its KV and compute,
    // rounded up to 17 GiB in `vram::COLD_RESERVE_LOCAL_LLM`. It was 24 GiB
    // while the default was the 16.09 GB `gemma4-31b-it`.
    //
    // Both are estimates: koharu measures the real figure during a run and it
    // does not exist before one. Lower this if a smaller `--llm` is being
    // refused -- and RAISE it if a larger one is being admitted and then dying,
    // because the reserve is one number rather than a per-model table and
    // `--llm gemma4-31b-it` wants roughly 3 GB more than the default assumes.
    /// VRAM, in bytes, that one cold page is assumed to need; a request that does not
    /// fit gets a 507 reply. Default: 21 GiB with --provider local, 4 GiB otherwise.
    #[arg(long, value_name = "BYTES")]
    pub cold_reserve_bytes: Option<u64>,

    /// Extra Origin values to accept verbatim, for diagnosing what the browser
    /// actually sends.
    #[arg(long = "allow-origin", value_name = "ORIGIN")]
    pub allow_origins: Vec<String>,

    /// Extra Host values to accept, for a deliberate non-loopback bind.
    #[arg(long = "allow-host", value_name = "HOST")]
    pub allow_hosts: Vec<String>,
}

pub struct Resolved {
    pub addr: SocketAddr,
    pub hyphenation: Option<koharu_renderer::HyphenationPolicy>,
    /// `None` means every balloon solves alone, as it always did.
    pub size_coherence: Option<f32>,
    /// Whether the renderer shrinks text it would otherwise set on top of other
    /// text. See `collision_relief`.
    pub collision_relief: bool,
    /// Whether a cut balloon's lettering anchors at the source ink's center
    /// rather than the visible part's. See `no_edge_anchored_lettering`.
    pub edge_anchored_lettering: bool,
    /// The story window, in source/target pairs. `0` is the no-story arm. Reaches
    /// `Stories` at construction and is never read again.
    pub story_pairs: usize,
    /// Whether an `onomatopoeia`-labelled region's pair is kept out of the
    /// story window. Read at the one record site in `routes.rs`, through the
    /// composed predicate `regions::feeds_story`.
    pub story_excludes_sfx: bool,
    pub cpu: bool,
    /// `None` means authentication is off.
    pub token_digest: Option<[u8; 32]>,
    /// Set only when the token was generated here, so it can be shown once.
    pub generated_token: Option<String>,
    pub pinned: Pinned,
    pub defaults: Defaults,
    pub providers: ProvidersConfig,
    pub allowed_origins: Vec<String>,
    pub allowed_hosts: Vec<String>,
    pub font_families: Vec<String>,
    pub max_upload_bytes: usize,
    pub queue_timeout: Duration,
    /// `None` disables the idle unload entirely.
    pub idle_unload: Option<Duration>,
    pub warmup: bool,
    /// Shut down when this process exits; `None` runs unwatched.
    pub watch_pid: Option<u32>,
    pub uppercase_dialogue: bool,
    /// Whether a region whose OCR text is not text is refused the page.
    pub skip_implausible_text: bool,
    /// Whether that refusal is scoped to the site's own text when the region is a
    /// watermark stamped across artwork. Carried at the top level as well as in
    /// `defaults`, because the LETTER side reads it here (`labels::hide_implausible`
    /// through `routes.rs`) and the ERASE side reads it off `ProcessorConfig`.
    /// The two must agree or a region is erased by one gate and refused by the
    /// other.
    pub scope_watermark_refusals: bool,
    /// The LETTER half of `--leave-misread-bubbles`, carried at the top level
    /// for the same two-sides reason as `scope_watermark_refusals` above; the
    /// erase half rides `defaults` into `ProcessorConfig`.
    pub leave_misread_bubbles: bool,
    /// The LETTER half of `--korean-script-strict`; same shape.
    pub korean_script_strict: bool,
    /// Whether the redundant half of a duplicate-lettered pair is refused the
    /// page. See `letter_duplicate_text`.
    pub skip_duplicate_text: bool,
    /// Whether that gate decides on the frames as drawn rather than on their
    /// bounding boxes. See `duplicate_oriented_overlap`.
    pub duplicate_oriented_overlap: bool,
    /// Whether that gate additionally requires the two SOURCE boxes to
    /// coincide. See `duplicate_shared_source`.
    pub duplicate_shared_source: bool,
    pub fit_free_text: bool,
    pub source_ink_fraction: f32,
    pub sfx_dictionary: crate::sfx::Dictionary,
    pub cold_reserve_bytes: u64,
}

impl Cli {
    pub fn resolve(&self, env_token: Option<String>) -> Result<Resolved> {
        let provider = models::parse_provider(&self.provider).map_err(bad_flag)?;
        let wire_provider =
            models::provider_wire_name(&self.provider).ok_or_else(|| anyhow!("unknown provider"))?;
        let llm = match (&self.llm, wire_provider) {
            (Some(model), _) => model.clone(),
            (None, models::PROVIDER_LOCAL) => models::DEFAULT_LOCAL_MODEL.to_owned(),
            // Ollama tags are specific to the user's installation, so
            // defaulting would send a name the server does not have.
            (None, _) => bail!(
                "--llm is required with --provider ollama: pass an Ollama model tag \
                 (run `ollama list` to see them). Koharu registry ids such as {} are only \
                 valid with --provider local.",
                models::DEFAULT_LOCAL_MODEL
            ),
        };

        // llama.cpp takes a negative temperature as "greedy" and an absurd one
        // without complaint, so a typo would otherwise surface as bad prose
        // rather than as an error.
        if let Some(temperature) = self.translation_temperature
            && !(0.0..=2.0).contains(&temperature)
        {
            bail!("--translation-temperature must be between 0.0 and 2.0, got {temperature}");
        }

        let hyphenation = models::parse_hyphenation(&self.hyphenation).map_err(bad_flag)?;

        // A ratio below 1.0 would ask for a ceiling *under* the prevailing size,
        // which is not a tighter setting but a different and much worse rule --
        // so it reads as "off" rather than being clamped into meaning something
        // the caller did not ask for. NaN takes the same branch.
        if self.size_coherence.is_finite() && self.size_coherence > 4.0 {
            bail!(
                "--size-coherence {} is looser than the 4:1 spread the rule exists to close; \
                 pass a value at or below 4.0, or below 1.0 to turn it off",
                self.size_coherence
            );
        }
        let size_coherence = (self.size_coherence >= 1.0).then_some(self.size_coherence);

        // Refused rather than clamped, for the same reason `--size-coherence`
        // refuses: silently running a different window than the one asked for is
        // the failure mode a measurement flag exists to remove. A negative needs
        // no check here -- clap rejects it while parsing the `usize`.
        if self.story_pairs > MAX_STORY_PAIRS {
            bail!(
                "--story-pairs {} is past the {MAX_STORY_PAIRS} the translation stage's measured \
                 KV headroom covers for {}; pass a value at or below {MAX_STORY_PAIRS}, or 0 to \
                 turn the story window off. That ceiling is the DEFAULT model's: a heavier --llm \
                 affords far less (gemma4-31b-it is already over its own headroom at 256 pairs on \
                 dense material) and nothing here checks that for you.",
                self.story_pairs,
                models::DEFAULT_LOCAL_MODEL
            );
        }

        // A negative fraction is a typo, and the `> 0.0` filter below would turn
        // it silently into "off" -- which reads as the ceiling working while it
        // is not there at all. `0` is the documented off switch, so say so
        // rather than accepting a second spelling of it. NaN is caught here too.
        if !(self.large_crop_ocr_max_area >= 0.0) {
            bail!(
                "--large-crop-ocr-max-area {} is not a fraction of the page area; pass a \
                 positive value, or 0 to turn the ceiling off",
                self.large_crop_ocr_max_area
            );
        }

        // `parse_ocr`, not `resolve_ocr`: the substitute is never substituted.
        // Its own errors, because `parse_ocr`'s names `--ocr` and offers the one
        // value this flag refuses.
        let ocr_substitute = match self.hunyuan_substitute.as_deref() {
            None => None,
            Some(value) => match models::parse_ocr(value) {
                Ok(koharu_pipeline::OcrModel::HunyuanOcr1_5) => bail!(
                    "--hunyuan-substitute names the engine that replaces hunyuan-ocr-1.5, so it \
                     cannot be hunyuan-ocr-1.5; pass paddleocr-vl-1.6 or another --ocr value"
                ),
                Ok(engine) => Some(engine),
                Err(_) => bail!(
                    "bad --hunyuan-substitute \"{value}\"; use paddleocr-vl-1.6, manga-ocr, \
                     baberu-ocr, ollama-vision"
                ),
            },
        };

        let defaults = Defaults {
            ocr: models::resolve_ocr(&self.ocr, ocr_substitute.as_ref()).map_err(bad_flag)?,
            ocr_substitute,
            inpainting: models::parse_inpainting(&self.inpainting, self.rorem_steps, self.flux_strength).map_err(bad_flag)?,
            target_language: models::parse_language(&self.target_language).map_err(bad_flag)?,
            // An explicit `--translation-instructions ""` means "send none", not
            // "send an empty label": `prompt.rs` appends "Additional
            // instructions: " unconditionally around whatever it is given.
            instructions: self
                .translation_instructions
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            temperature: self.translation_temperature,
            // The clap shape is `num_args = 0..=1` with NO `default_value`, so
            // an absent flag really is `None` and this `unwrap_or` really is
            // the decision -- the same trap note as `read_textless_bubbles`.
            containment_clause: self.containment_clause.unwrap_or(false),
            // OFF -- a prompt change re-rolls every page, so it ships
            // selectable rather than on.
            segment_context: self.segment_context.unwrap_or(false),
            swa_full: self.swa_full,
            translate_sfx: !self.no_translate_sfx,
            refine_text_mask: self.refine_text_mask,
            rorem_steps: self.rorem_steps,
            flux_strength: self.flux_strength,
            onomatopoeia_threshold: self.onomatopoeia_threshold,
            mask_scale: self.mask_scale,
            large_crop_ocr_px: self.large_crop_ocr.filter(|px| *px > 0),
            // Zero is "off", matching `--large-crop-ocr 0`, and the negated `>`
            // sends NaN down the same branch. NaN must never reach the config:
            // `PipelineConfig` derives `PartialEq` and `needs_reload` compares
            // it, so a NaN would make every request unequal to the applied
            // config and reload the pipeline on every page.
            large_crop_ocr_max_area: Some(self.large_crop_ocr_max_area)
                .filter(|fraction| *fraction > 0.0),
            // Flattened for the same reason as `skip_empty_stages` below: the
            // region is either read or it is not, so there is no third state for
            // an `Option` to carry into `Defaults`.
            skip_implausible_regions: self.skip_implausible_regions.unwrap_or(true),
            // Default lives in the `unwrap_or` and NOT in a clap `default_value`,
            // so that flipping it here really does flip the shipped behaviour.
            // Some of `cli.rs`'s `<BOOL>` flags are written the other way and
            // the `unwrap_or` beside them is unreachable -- clap's
            // `default_value` fills the option before it gets here.
            rotate_free_text_columns: self.rotate_free_text_columns.unwrap_or(true),
            /* ON, after BOTH arms were rendered -- the same bar its sibling
             * above was held to.
             *
             * What was seen, on one slice of a Chinese test chapter:
             *   OFF  the artist's brush calligraphy ERASED and an invented name
             *        lettered over the hole, off a Japanese-shaped misread on a
             *        Chinese page.
             *   ON   the calligraphy untouched. The grown box is past
             *        `CROSS_SLICE_HEIGHT_SHARE`, so the cross-slice height gate
             *        refuses it before OCR: 1 region, 0 lettered.
             *
             * The trade this weighs -- a confident wrong string
             * against the artist's Chinese left standing -- resolved to the second
             * on every arm measured, joined and unjoined alike. Chapter-wide the
             * flip costs nothing: LOST 0 over 178 boundaries, and 0 edge hints at
             * all over 221 manga pages. */
            repair_clipped_columns: self.repair_clipped_columns.unwrap_or(true),
            /* ON, on rendered evidence.
             *
             * Earlier arms each broke something: the first destroyed the balloon
             * (79.7% of its white body erased), and the second fixed that but
             * still lettered an em dash across a character's face on a webtoon
             * test page.
             *
             * What changed is that EVERY page in the corpus this rule fires on has
             * been rendered, rather than predicted. Three fixes did it:
             *   - a read of nothing but prolongation marks is refused -- that
             *     page was OCR reading the character's own mouth line as `ー`
             *   - an ink-fraction floor from a census of 2,519 bubbles over
             *     773 pages, which rejects that page's spurious bubble outright
             *   - a synthesised region marks itself, so a punctuation-only
             *     read is not lettered across the ink it just spared
             *
             * On the ten pages it fires on: one correct win (the reported
             * exhibit), one correctly lettered (`ハァ・・・`, real kana), one
             * refused as a duplicate, seven no-ops, and ZERO defects.
             *
             * **Three predictions were refuted by renders on the way here**, each
             * after the counters agreed with them: the ink floor was expected to
             * gate the dash page and did not; a character-count rule looked sound
             * until a render found a real balloon holding one character; and the
             * prolongation-mark refusal looked complete until a render showed the
             * dash still drawn. Do not flip this back on an argument -- flip it on
             * a render. */
            read_textless_bubbles: self.read_textless_bubbles.unwrap_or(true),
            // `read_textless_bubbles`'s default lives in its `unwrap_or` above and NOT in a
            // clap `default_value`, or that `unwrap_or` becomes unreachable and a prove-red
            // on it reads as a blind test rather than a wired one.
            // 0.20, adopted with the tie-break on the two-corpus census. `0` is
            // the off arm rather than a
            // floor that admits everything, mirroring `--large-crop-ocr 0`.
            // The band it opens is replacement-only, so without the tie-break
            // beside it this is inert -- proven byte-identical over the whole
            // chapter (A1 vs A0: 339 of 339 images, 0.000%).
            joined_page_text_floor: match self.joined_page_text_floor {
                None => Some(0.20),
                Some(floor) if floor > 0.0 => Some(floor),
                Some(_) => None,
            },
            // ON, on the rendered census -- and SCOPED to declared
            // Chinese/Korean in `desired_config`, which is where the ja arm's
            // refutation is recorded. The default lives HERE, not in a clap
            // `default_value`.
            axis_aware_nms: self.axis_aware_nms.unwrap_or(true),
            /* ON, on a whole-chapter render read page by page, which came out
             * better on every page but three panels. The default lives HERE,
             * not in a clap `default_value`.
             *
             * PRICED ON A TWO-FLAG ARM, which is the caveat to carry: that
             * render moved this flag and `--strike-through-devices` together, so
             * no part of the verdict is attributable to either flag alone. One
             * page is the proof it matters -- byte-identical regions AND
             * `rendered_text`, yet 11,242 differing pixels, 10,096 of them newly
             * crimson from the OTHER flag, on a page where this one changed
             * nothing. `false` is still a byte-exact control arm. */
            nms_residue_regions: self.nms_residue_regions.unwrap_or(true),
            /* ON, on a one-flag render read against the official English
             * edition as the standard -- close to it rather than 1:1.
             *
             * The measurement behind it: **0 of 143 lettered regions lost**,
             * 331 regions with 188 refused and 143 lettered in BOTH arms, seam
             * decisions byte-identical, and 44/24/31 to this arm over 100 blind
             * reworded pairs. What it does NOT rest on is efficiency -- segments
             * fall 22% but prompt tokens only 1.6%, because the story window
             * dominates the prompt.
             *
             * Turning it on RE-ROLLS translations chapter-wide, once. `false` is
             * still a byte-exact control arm and reverts this in one word. */
            skip_unlettered_reads: self.skip_unlettered_reads.unwrap_or(true),
            /* ON, on the same whole-chapter render as `nms_residue_regions`
             * above, and read with that flag's two-flag caveat. The default
             * lives HERE, not in a clap `default_value`.
             *
             * The detector is still validated on n=2 positives. The rendered
             * chapter was accepted anyway; a thin validation set is a reason
             * to WATCH this flag, not to hold it back. `false` reverts it in
             * one word. */
            strike_through_devices: self.strike_through_devices.unwrap_or(true),
            // ON -- the styling's shipping arm. The default lives HERE, not in
            // a clap `default_value`; `false` is the same-binary control arm
            // the A/B renders on.
            sampled_ink_lettering: self.sampled_ink_lettering.unwrap_or(true),
            /* ON, on the renders.
             *
             * Both arms, one run each. A joined five-slice column goes from the
             * artist's Chinese standing untranslated to `BLUE FLAME, FROST,
             * GALE` lettered down its own axis, all three names correct. The
             * safety case is measured rather than argued: six manga pages of
             * upright vertical Japanese render BYTE-IDENTICAL between the
             * arms, so the kana census provably never reaches the population
             * this could have broken. */
            reread_rotated_columns: self.reread_rotated_columns.unwrap_or(true),
            /* ON, on the chapter render: 14 recoveries, 8 pages changed, every
             * one looked at. */
            upright_pass: self.upright_pass.unwrap_or(true),
            /* ON, on its render: one page changed of 335, the balloon intact,
             * controls byte-identical, cost inside noise. */
            flip_reread_bubbles: self.flip_reread_bubbles.unwrap_or(true),
            /* ON, each on its own render. The lettering and the erase are
             * SEPARATE flags so a render can adjudicate them separately. */
            spot_rescue: self.spot_rescue.unwrap_or(true),
            spot_rescue_erase: self.spot_rescue_erase.unwrap_or(true),
            // ON, on the rendered A/Bs: one fire in 246 slices, byte-identity
            // everywhere else.
            replace_scream_marks: self.replace_scream_marks.unwrap_or(true),
            // ON, on its own render: with the paint-back gate in place it
            // changes exactly 2 of 179 reader-final slices and delivers the
            // joined column.
            spot_rescue_joined: self.spot_rescue_joined.unwrap_or(true),
            // 0.17: the value the chapter was rendered at (3 of 179 pages
            // change, rescue population byte-identical). `None` is no
            // longer reachable from the CLI; the off arm is an explicit `inf`,
            // which no finite confidence gap can clear -- 0.0 is NOT off, it
            // flips ties (pinned in `choose_orientation`'s tests).
            orientation_confidence_margin: self.orientation_confidence_margin.or(Some(0.17)),
            // Same: `None` IS the shipping value and means "do not re-read".
            perturb_reread_grow_px: self.perturb_reread_grow_px,
            /* ON, on the rendered cost: sound effects lettered sideways are
             * acceptable.
             *
             * Both arms rendered. Three display columns go from a fit box
             * 2.29x / 2.12x / 1.40x their own ink to EXACTLY 1.00, so the English
             * lands inside the area the eraser already cleaned. The price is sound
             * effects on manga lettered sideways -- `WH-`/`WIN` on one page and
             * `*TAP*` on another, 0.169% and 0.046% of their pages. Dialogue is untouched at
             * any setting: balloon text takes the bubble path and never reaches
             * here, which is what bounds the cost to effects. */
            turn_unjoined_columns: self.turn_unjoined_columns.unwrap_or(true),
            /* ON, same decision, same renders, and the two only make sense
             * together: turned-but-unscoped reads the column correctly and then
             * throws it away with the watermark, and scoped-but-unturned letters a
             * confident wrong name.
             *
             * Cost, stated: a rescued region is erased and inpainted WHOLE, plate
             * included. Bounded to regions carrying both -- of the manhua controls
             * only one page moved, from refused to a correctly read character
             * name, which is the one genuine loss of story text in that chapter. */
            scope_watermark_refusals: self.scope_watermark_refusals.unwrap_or(true),
            // The erase halves of the paired misread levers. ON, on a read of
            // the censused chapters; each reads the SAME option as its
            // letter-side copy below,
            // including the default, so the two ends cannot disagree -- flip
            // both or neither.
            leave_misread_bubbles: self.leave_misread_bubbles.unwrap_or(true),
            korean_script_strict: self.korean_script_strict.unwrap_or(true),
            // ON, on the rendered evidence (one page changed, a recovered groan
            // lettered in English, nothing else moved) -- this wires the
            // reserve engine into the shipping path.
            reread_refused_dialogue: self.reread_refused_dialogue.unwrap_or(true),
            // Flattened the same way. ON on a pixel comparison; see the flag's
            // own doc.
            skip_implausible_masks: self.skip_implausible_masks.unwrap_or(true),
            // Defaulted ON, and the reason it may be where its sibling may not is
            // that the pixels exist: the birds on 381206/131 survive with it and
            // do not without it.
            withdraw_illegible_masks: self.withdraw_illegible_masks.unwrap_or(true),
            withdraw_unread_masks: self.withdraw_unread_masks.unwrap_or(true),
            seam_safe_erase: self.seam_safe_erase.unwrap_or(true),
            ink_mask: self.ink_mask,
            release_cached_vram: self.release_cached_vram,
            // Flattened here rather than carried as an `Option` the way
            // `release_cached_vram` is: `ProcessorConfig` holds a plain `bool`,
            // because "leave it to the model" is not a third state -- a stage
            // either gets asked or it does not.
            skip_empty_stages: self.skip_empty_stages.unwrap_or(true),
            debug_mask_dir: self.debug_mask_dir.clone(),
        };

        let (token_digest, generated_token) = self.resolve_token(env_token);

        if token_digest.is_none() {
            if !self.addr.ip().is_loopback() {
                bail!(
                    "refusing to bind {} without authentication: drop --no-token, or bind a \
                     loopback address",
                    self.addr
                );
            }
            if let Some(origin) = self.allow_origins.iter().find(|origin| is_web_origin(origin)) {
                bail!(
                    "--no-token with --allow-origin {origin} would let any web page drive this \
                     server: keep the token, or drop the allowed origin"
                );
            }
        }

        let mut providers = ProvidersConfig::default();
        // The default already points at Ollama; only override when asked.
        if let Some(base_url) = self.base_url.clone() {
            providers.openai_compatible = OpenAiCompatibleConfig {
                base_url: Some(base_url),
            };
        }

        let mut allowed_hosts = allowed_hosts_from(self.addr);
        for host in &self.allow_hosts {
            if !allowed_hosts.contains(host) {
                allowed_hosts.push(host.clone());
            }
        }

        Ok(Resolved {
            addr: self.addr,
            hyphenation,
            size_coherence,
            // Inverted here, once, like `skip_implausible_text`, so downstream
            // reads as "is the pass on?" rather than a double negative.
            collision_relief: !self.no_collision_relief,
            edge_anchored_lettering: !self.no_edge_anchored_lettering,
            story_pairs: self.story_pairs,
            /* ON, on the full-chapter A/B. The measurement behind it: the
             * exhibit page letters the ability name as a name against the
             * control's "BOOM BOOM BOOM BOOM!", with the licensed edition
             * confirming the name class; 74/142 live regions reworded and
             * 45/178 pages differing -- roughly HALF `--segment-context`'s
             * blast radius -- with every term that lever lost kept stable.
             * The one recorded regression candidate: a drawn word-glyph
             * letters as a sound instead of its meaning, because the SFX's
             * own request also loses the window pairs.
             *
             * Turning it on re-rolls translations from the first excluded pair
             * onward, once. `false` is a byte-exact control arm and reverts
             * this in one word. The default lives HERE, not in a clap
             * `default_value`. */
            story_excludes_sfx: self.story_excludes_sfx.unwrap_or(true),
            cpu: self.cpu,
            token_digest,
            generated_token,
            pinned: Pinned {
                provider,
                llm,
                wire_provider,
            },
            defaults,
            providers,
            allowed_origins: self.allow_origins.clone(),
            allowed_hosts,
            font_families: if self.font_families.is_empty() {
                DEFAULT_FONT_FAMILIES.map(str::to_owned).to_vec()
            } else {
                self.font_families.clone()
            },
            max_upload_bytes: self.max_upload_bytes,
            queue_timeout: Duration::from_secs(self.request_timeout_secs),
            idle_unload: (self.idle_unload_secs > 0)
                .then(|| Duration::from_secs(self.idle_unload_secs)),
            warmup: self.warmup,
            watch_pid: self.watch_pid,
            uppercase_dialogue: self.uppercase_dialogue,
            // Inverted here, once, so every reader downstream asks the positive
            // question ("is the gate on?") rather than a double negative.
            skip_implausible_text: !self.letter_implausible_text,
            // Read from the SAME option the `defaults` copy reads, so the letter
            // side and the erase side can never disagree -- including the default,
            // which is why this `unwrap_or` must be flipped with the other one.
            scope_watermark_refusals: self.scope_watermark_refusals.unwrap_or(true),
            // Same same-option contract for the misread pair: these letter-side
            // copies and the `defaults` erase-side ones above read one Option
            // each, defaults included, so a flip moves both ends together.
            // ON, with the erase-side copies above.
            leave_misread_bubbles: self.leave_misread_bubbles.unwrap_or(true),
            korean_script_strict: self.korean_script_strict.unwrap_or(true),
            skip_duplicate_text: !self.letter_duplicate_text,
            duplicate_oriented_overlap: !self.no_duplicate_oriented_overlap,
            // ON, on a rendered A/B crop: additive corroboration, twin
            // authored marks both letter.
            duplicate_shared_source: self.duplicate_shared_source.unwrap_or(true),
            fit_free_text: !self.no_fit_free_text,
            source_ink_fraction: self.source_ink_fraction,
            sfx_dictionary: self.load_sfx_dictionaries()?,
            cold_reserve_bytes: self
                .cold_reserve_bytes
                .unwrap_or_else(|| vram::default_cold_reserve(provider)),
        })
    }

    /// Reads every `--sfx-dictionary` in order. A missing or malformed file is
    /// a startup error rather than a warning: the flag was asked for
    /// explicitly, and silently lettering a page without the pins the operator
    /// requested is the failure that would go unnoticed.
    fn load_sfx_dictionaries(&self) -> Result<crate::sfx::Dictionary> {
        let mut dictionary = crate::sfx::Dictionary::default();
        for path in &self.sfx_dictionaries {
            let json = std::fs::read_to_string(path)
                .map_err(|error| anyhow!("cannot read --sfx-dictionary {}: {error}", path.display()))?;
            let added = dictionary
                .merge_json(&json)
                .map_err(|error| anyhow!("bad --sfx-dictionary {}: {error}", path.display()))?;
            tracing::info!(file = %path.display(), added, total = dictionary.len(), "sfx dictionary loaded");
        }
        Ok(dictionary)
    }

    /// `--store-dir`, else `env` (BIRELATE_STORE_DIR; empty counts as unset),
    /// made absolute because `Store::configure` refuses a relative root. `None`
    /// leaves Koharu's default store alone. The environment is an argument for
    /// the same reason `resolve`'s token is: so the precedence is testable.
    pub fn resolve_store_dir(
        &self,
        env: Option<std::ffi::OsString>,
    ) -> Result<Option<std::path::PathBuf>> {
        let Some(dir) = self
            .store_dir
            .clone()
            .or_else(|| env.filter(|value| !value.is_empty()).map(Into::into))
        else {
            return Ok(None);
        };
        std::path::absolute(&dir)
            .map(Some)
            .map_err(|error| anyhow!("bad --store-dir {}: {error}", dir.display()))
    }

    fn resolve_token(&self, env_token: Option<String>) -> (Option<[u8; 32]>, Option<String>) {
        if self.no_token {
            return (None, None);
        }
        let configured = self
            .token
            .as_deref()
            .or(env_token.as_deref())
            .map(str::trim)
            .filter(|token| !token.is_empty());
        match configured {
            Some(token) => (Some(digest(token)), None),
            None => {
                // 32 hex characters: short enough to retype into the popup,
                // long enough that guessing it is not a threat.
                let token = uuid::Uuid::new_v4().simple().to_string();
                (Some(digest(&token)), Some(token))
            }
        }
    }
}

/// An Origin a web page could actually present. `null` counts: a cross-origin
/// request that suppresses its origin arrives exactly that way.
fn is_web_origin(origin: &str) -> bool {
    origin == "null" || origin.starts_with("http://") || origin.starts_with("https://")
}

fn bad_flag(error: crate::error::ApiError) -> anyhow::Error {
    anyhow!("{}", error.message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli(extra: &[&str]) -> Cli {
        let mut argv = vec!["birelate-server"];
        argv.extend_from_slice(extra);
        Cli::try_parse_from(argv).unwrap()
    }

    /// ON by default, which no collision gate before it was, and turned off by a
    /// flag rather than on by one. A five-judge blind panel put 20 of 21 arm
    /// preferences behind it with 4 of 4 controls correctly called identical.
    #[test]
    fn collision_relief_is_on_by_default_and_the_flag_turns_it_off() {
        assert!(cli(&[]).resolve(None).unwrap().collision_relief);
        assert!(
            !cli(&["--no-collision-relief"])
                .resolve(None)
                .unwrap()
                .collision_relief
        );
    }

    /// The anchored arm ships **ON**, on the adjudicated A/B render, after
    /// first shipping OFF. The test survives the flip with its polarity
    /// reversed, which is the whole point of having written it: the next move
    /// of this default also has to be
    /// deliberate.
    #[test]
    fn edge_anchored_lettering_is_on_by_default_and_the_flag_turns_it_off() {
        assert!(cli(&[]).resolve(None).unwrap().edge_anchored_lettering);
        assert!(
            !cli(&["--no-edge-anchored-lettering"])
                .resolve(None)
                .unwrap()
                .edge_anchored_lettering
        );
    }

    /// The oriented arm ships **ON**, and this pins it there.
    ///
    /// **This test once asserted the opposite, deliberately.** It shipped
    /// OFF on two pages of evidence, with this test written to make
    /// flipping the default deliberate rather than incidental. It was flipped
    /// on the 219-page corpus measurement in
    /// `no_duplicate_oriented_overlap`'s doc comment. The test survives the flip
    /// with its polarity reversed, which is the whole point of having written it:
    /// the next move of this default also has to be deliberate.
    #[test]
    fn the_oriented_duplicate_geometry_is_on_by_default_and_the_flag_turns_it_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().duplicate_oriented_overlap,
            "shipping default must be the drawn-frame geometry"
        );
        assert!(
            !cli(&["--no-duplicate-oriented-overlap"])
                .resolve(None)
                .unwrap()
                .duplicate_oriented_overlap,
            "the A/B escape hatch must reach the bounding-box geometry"
        );
        // And it is independent of the gate itself: turning the gate off does not
        // silently turn this off, nor the reverse. Both directions, because the
        // pair is what a caller actually configures -- asserting each flag alone
        // would pass with the two silently wired together.
        let gate_off = cli(&["--letter-duplicate-text"]).resolve(None).unwrap();
        assert!(!gate_off.skip_duplicate_text);
        assert!(gate_off.duplicate_oriented_overlap);

        let both_off = cli(&["--letter-duplicate-text", "--no-duplicate-oriented-overlap"])
            .resolve(None)
            .unwrap();
        assert!(!both_off.skip_duplicate_text);
        assert!(!both_off.duplicate_oriented_overlap);
    }

    /// The shared-source corroboration SHIPS ON; `false` is the same-binary
    /// control arm, independently of the gate and the geometry. It was flipped
    /// ON on the rendered A/B.
    #[test]
    fn the_shared_source_arm_ships_on_and_false_is_the_control_arm() {
        assert!(
            cli(&[]).resolve(None).unwrap().duplicate_shared_source,
            "ON by default, on the rendered A/B crop"
        );
        let off = cli(&["--duplicate-shared-source", "false"])
            .resolve(None)
            .unwrap();
        assert!(!off.duplicate_shared_source, "false is the same-binary control arm");
        // Independent of its neighbours: the control arm does not drag the gate
        // or the geometry, and both keep their own defaults.
        assert!(off.skip_duplicate_text);
        assert!(off.duplicate_oriented_overlap);
    }

    #[test]
    fn size_coherence_is_on_by_default_and_turned_off_by_a_ratio_below_one() {
        let default = cli(&[]).resolve(None).unwrap().size_coherence;
        assert_eq!(default, Some(1.25));

        /* Off has to be reachable, and it has to be reachable from the same
         * flag: the two arms of an A/B that needs a rebuild to switch is the
         * shape that let a stale release binary ignore a whole feature for an
         * entire measurement run. */
        for off in ["0", "0.5"] {
            assert_eq!(
                cli(&["--size-coherence", off]).resolve(None).unwrap().size_coherence,
                None,
                "--size-coherence {off}"
            );
        }
        // A negative needs the `=` form -- clap reads a bare `-1` as a flag --
        // which is exactly why `0` is the documented way to turn it off.
        assert_eq!(
            cli(&["--size-coherence=-1"]).resolve(None).unwrap().size_coherence,
            None
        );
        assert_eq!(
            cli(&["--size-coherence", "1"]).resolve(None).unwrap().size_coherence,
            Some(1.0),
            "exactly 1.0 is the tightest setting, not the off switch"
        );
        // Looser than the defect is a typo, not a preference.
        assert!(cli(&["--size-coherence", "10"]).resolve(None).is_err());
    }

    /// 96 is the shipping window and the flag must not have moved it. The whole
    /// point of the flag is that a 24/48/96 comparison stops needing a rebuild --
    /// which is worthless if merely adding it changed the arm everything measured
    /// so far was measured on.
    /// The SFX window exclusion ships **ON**, on the full-chapter A/B, after
    /// first shipping OFF. The test
    /// survives the flip with its polarity reversed, which is the whole point
    /// of asserting through `Cli::resolve`: same clap shape as the flags above
    /// -- `num_args = 0..=1` and **no** `default_value` -- so the
    /// `unwrap_or(true)` really is the decision, and a green after flipping it
    /// back would mean a `default_value` sneaked onto the attribute.
    #[test]
    fn the_sfx_window_exclusion_ships_on_and_false_is_the_control_arm() {
        assert!(
            cli(&[]).resolve(None).unwrap().story_excludes_sfx,
            "ON by default, on the full-chapter A/B"
        );
        assert!(
            cli(&["--story-excludes-sfx"])
                .resolve(None)
                .unwrap()
                .story_excludes_sfx,
            "the bare flag still spells it out"
        );
        assert!(
            !cli(&["--story-excludes-sfx", "false"])
                .resolve(None)
                .unwrap()
                .story_excludes_sfx,
            "false is the byte-exact control arm, and reverts the flip in one word"
        );
    }

    #[test]
    fn the_story_window_is_ninety_six_unless_asked_otherwise() {
        assert_eq!(cli(&[]).resolve(None).unwrap().story_pairs, 96);
        assert_eq!(crate::story::DEFAULT_PAIRS, 96);
        assert_eq!(
            cli(&["--story-pairs", "24"]).resolve(None).unwrap().story_pairs,
            24
        );
    }

    /// Zero means "no story window", not "an empty one that still books a slot".
    /// It is the no-story arm, and it has to live on the server: the extension
    /// always sends a story id, so withholding one is not something a measurement
    /// run can ask the browser for.
    #[test]
    fn a_zero_story_window_is_the_no_story_arm_rather_than_an_error() {
        assert_eq!(cli(&["--story-pairs", "0"]).resolve(None).unwrap().story_pairs, 0);
    }

    /// Past the ceiling the value is refused, not clamped. A window silently
    /// replaced by a different one is precisely what a flag whose only purpose is
    /// measurement must never do.
    #[test]
    fn an_unaffordable_story_window_is_refused_with_the_ceiling_named() {
        // Named rather than merely compared to itself, for the same reason the
        // default model is: `MAX_STORY_PAIRS.to_string()` passes at any value.
        // 1,024 is the MoE's number -- see the flag's doc comment -- and it was
        // 256 while the incumbent was the default.
        assert_eq!(MAX_STORY_PAIRS, 1024);
        assert_eq!(
            cli(&["--story-pairs", "256"]).resolve(None).unwrap().story_pairs,
            256,
            "the old ceiling stays reachable, so the incumbent's arm can be re-run"
        );
        assert_eq!(
            cli(&["--story-pairs", &MAX_STORY_PAIRS.to_string()])
                .resolve(None)
                .unwrap()
                .story_pairs,
            MAX_STORY_PAIRS,
            "the ceiling itself is allowed"
        );
        // `.err().expect(..)` rather than `unwrap_err()`: the latter needs
        // `Resolved: Debug`, which nothing else in this crate requires.
        let error = cli(&["--story-pairs", &(MAX_STORY_PAIRS + 1).to_string()])
            .resolve(None)
            .err()
            .expect("a window past the KV allowance must be refused")
            .to_string();
        assert!(error.contains("--story-pairs"), "{error}");
        assert!(error.contains("0 to turn"), "{error}");
    }

    #[test]
    fn ollama_requires_an_explicit_model() {
        assert!(cli(&["--provider", "ollama"]).resolve(None).is_err());
    }

    #[test]
    fn local_defaults_to_the_koharu_registry_model() {
        let resolved = cli(&["--provider", "local"]).resolve(None).unwrap();
        assert_eq!(resolved.pinned.llm, models::DEFAULT_LOCAL_MODEL);
        assert_eq!(resolved.pinned.wire_provider, models::PROVIDER_LOCAL);
        // Spelled out as well as compared symbolically. Comparing `resolved` to
        // the const alone is a tautology -- it passes whatever the const says --
        // and the const's value is what the launcher and the popup's placeholder
        // copy by hand. See the const's own comment for why it is this model.
        assert_eq!(models::DEFAULT_LOCAL_MODEL, "gemma4-26b-a4b-it");
    }

    /// The incumbent is a supported value, not a removed one. The whole switch
    /// rests on a null result, so the arm it was measured against has to stay
    /// runnable -- otherwise nobody can reproduce the comparison that chose it.
    #[test]
    fn the_previous_default_is_still_reachable() {
        let resolved = cli(&["--llm", "gemma4-31b-it"]).resolve(None).unwrap();
        assert_eq!(resolved.pinned.llm, "gemma4-31b-it");
        assert_eq!(resolved.pinned.wire_provider, models::PROVIDER_LOCAL);
    }

    #[test]
    fn an_unauthenticated_server_may_not_leave_loopback() {
        assert!(
            cli(&["--no-token", "--addr", "0.0.0.0:8765"])
                .resolve(None)
                .is_err()
        );
        assert!(cli(&["--addr", "0.0.0.0:8765"]).resolve(None).is_ok());
    }

    #[test]
    fn an_unauthenticated_server_may_not_also_open_its_origin_policy() {
        assert!(
            cli(&["--no-token", "--allow-origin", "null"])
                .resolve(None)
                .is_err()
        );
        assert!(
            cli(&["--no-token", "--allow-origin", "https://example.test"])
                .resolve(None)
                .is_err()
        );
        // A token makes the same combination the user's own call to make.
        assert!(
            cli(&["--token", "s3cret", "--allow-origin", "null"])
                .resolve(None)
                .is_ok()
        );
    }

    #[test]
    fn the_flag_beats_the_environment() {
        let flag = cli(&["--token", "from-flag"])
            .resolve(Some("from-env".to_owned()))
            .unwrap();
        assert_eq!(flag.token_digest, Some(digest("from-flag")));
        assert!(flag.generated_token.is_none());

        let environment = cli(&[]).resolve(Some("from-env".to_owned())).unwrap();
        assert_eq!(environment.token_digest, Some(digest("from-env")));
        assert!(environment.generated_token.is_none());
    }

    #[test]
    fn a_token_is_generated_when_none_was_given() {
        let first = cli(&[]).resolve(None).unwrap();
        let second = cli(&[]).resolve(None).unwrap();
        let token = first.generated_token.clone().unwrap();
        assert_eq!(token.len(), 32);
        assert!(token.chars().all(|character| character.is_ascii_hexdigit()));
        assert_ne!(token, second.generated_token.unwrap());
        assert_eq!(first.token_digest, Some(digest(&token)));
    }

    #[test]
    fn no_token_really_means_no_token() {
        let resolved = cli(&["--no-token"]).resolve(Some("ignored".to_owned())).unwrap();
        assert!(resolved.token_digest.is_none());
        assert!(resolved.generated_token.is_none());
    }

    #[test]
    fn the_defaults_match_the_extensions_expectations() {
        let resolved = cli(&[]).resolve(None).unwrap();
        assert_eq!(resolved.addr, "127.0.0.1:8765".parse().unwrap());
        assert_eq!(resolved.max_upload_bytes, 33_554_432);
        assert_eq!(resolved.queue_timeout, Duration::from_secs(600));
        assert_eq!(resolved.idle_unload, Some(Duration::from_secs(300)));
        assert!(!resolved.warmup);
        assert_eq!(resolved.font_families, ["CCWildWords", "Arial"]);
        // Uppercasing stays OFF even though the face is now a comic one. See
        // the flag's own comment: CCWildWords maps upper and lower case to two
        // different capital designs, so `to_uppercase` would erase the
        // variation rather than add it.
        assert!(!resolved.uppercase_dialogue);
        // Drawn sound effects are lettered by default. They were detected,
        // folded into the erase mask and never painted back for the whole life
        // of this project, at about 2.2 per page.
        assert!(resolved.defaults.translate_sfx);

        // The three engine defaults the extension also hard-codes, and the whole
        // reason this test carries that name. Without this assertion a launcher
        // can sit on a stale engine after this default moves: nothing else
        // compares the two copies.
        //
        // These are not cosmetic. A per-request value that differs from the one
        // the server started with is a structural `!=` over the whole
        // `PipelineConfig` (`engine::needs_reload`), so it forces
        // `Pipeline::reload` on the first translate -- which builds a fresh
        // `Residency` and throws away every learned per-stage memory profile.
        //
        // If you change one of these, change `extension/background.js` DEFAULTS,
        // `extension/popup.js` DEFAULTS, `extension/popup.html`'s <option> order
        // and `scripts\serve.ps1` in the same commit. This assertion is the only
        // thing that will tell you that you forgot.
        assert_eq!(
            crate::models::ocr_model_name(&resolved.defaults.ocr),
            "hunyuan-ocr-1.5"
        );
        assert_eq!(
            crate::models::inpainting_model_name(&resolved.defaults.inpainting),
            "lama"
        );
        assert_eq!(resolved.defaults.target_language.tag(), "en-US");

        // The translator is on that list too. It behaves slightly
        // differently from the three above -- the extension sends `llm: ""` and
        // lets the server decide, so a mismatch does NOT force a reload -- but
        // the popup's placeholder and `scripts\serve.ps1` print or pass this
        // name, and a launcher passing an old one pins a model the user did
        // not choose.
        assert_eq!(resolved.pinned.llm, "gemma4-26b-a4b-it");
        assert_eq!(resolved.pinned.wire_provider, "local");
    }

    #[test]
    fn sound_effect_lettering_can_be_turned_off_for_the_a_b() {
        let resolved = cli(&["--no-translate-sfx"]).resolve(None).unwrap();
        assert!(!resolved.defaults.translate_sfx);
    }

    /// Mask refinement stays OFF by default: it costs ~1.5s a page and its only
    /// measurement so far found 0.19% of pixels. The flag is what makes
    /// re-measuring it against erased sound effects possible at all.
    #[test]
    fn mask_refinement_is_off_until_asked_for() {
        assert!(!cli(&[]).resolve(None).unwrap().defaults.refine_text_mask);
        assert!(
            cli(&["--refine-text-mask"])
                .resolve(None)
                .unwrap()
                .defaults
                .refine_text_mask
        );
    }

    /// Returning torch's cached VRAM is ON by default, and unlike the other
    /// defaults here that is *not* yet backed by a measurement -- it is on
    /// because the defect it addresses is a permanent latch that more than
    /// doubles page time, and a fix that has to be remembered as a flag will be
    /// forgotten. The off arm exists so the claim can be checked rather than
    /// believed.
    #[test]
    fn returning_cached_vram_is_on_by_default_and_can_be_turned_off() {
        assert_eq!(
            cli(&[]).resolve(None).unwrap().defaults.release_cached_vram,
            Some(true)
        );
        assert_eq!(
            cli(&["--release-cached-vram", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .release_cached_vram,
            Some(false)
        );
    }

    /// The empty-stage gate is ON by default. It can only ever skip a stage that
    /// `process` would have found empty anyway -- both sides run the same walk --
    /// so there is no arm where it changes a rendered page, and 29% of a measured
    /// webtoon chapter was cold LLM loads for slices with no text on them. The
    /// off arm exists so the behaviour change can be A/B'd, and so a
    /// measurement can still spell the old order.
    #[test]
    fn the_empty_stage_gate_is_on_by_default_and_can_be_turned_off() {
        assert!(cli(&[]).resolve(None).unwrap().defaults.skip_empty_stages);
        assert!(
            !cli(&["--skip-empty-stages", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .skip_empty_stages
        );
    }

    /// The column turn is ON by default, and the default lives in the `unwrap_or`.
    ///
    /// **Asserted rather than assumed, because `cli.rs` gets this wrong
    /// elsewhere.** Every `#[arg(..., default_value = "true")]` flag on an
    /// `Option<bool>` in this file makes clap fill in `Some(true)` when the flag
    /// is absent, so the `.unwrap_or(true)` beside it is unreachable and flipping
    /// it changes nothing -- a deliberately broken assertion on such a line once
    /// left the suite 239 green. This flag
    /// carries `default_missing_value` and **no** `default_value`, so an absent
    /// flag really is `None` and the `unwrap_or` really is the decision. Both arms
    /// are pinned so a future refactor cannot quietly move it.
    #[test]
    fn the_column_turn_is_on_by_default_and_can_be_turned_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.rotate_free_text_columns,
            "the turn ships ON, on the render"
        );
        assert!(
            !cli(&["--rotate-free-text-columns", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .rotate_free_text_columns,
            "the off arm has to restore the widened cell, or the A/B is gone"
        );
    }

    /// **The text-less-bubble read ships ON, and both arms are pinned.**
    ///
    /// An unpinned default can move without anyone noticing; this one is
    /// pinned for the reason the column-turn test above gives.
    ///
    /// The same clap trap applies as above: this flag carries `num_args = 0..=1`
    /// and **no** `default_value`, so an absent flag really is `None` and the
    /// `unwrap_or` really is the decision. The OFF arm is asserted too, because it
    /// is the one a reader falls back to if this turns out wrong on a page nobody
    /// rendered -- and because a flag that cannot be turned off is not a flag.
    #[test]
    fn textless_bubbles_are_read_by_default_and_can_be_turned_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.read_textless_bubbles,
            "ships ON, after every page in the corpus it fires on had been rendered"
        );
        assert!(
            !cli(&["--read-textless-bubbles", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .read_textless_bubbles,
            "the off arm is the fallback if this is ever wrong on an unrendered page"
        );
    }

    /// OFF is the shipped arm, and BOTH arms must parse, because the flag
    /// exists for a same-binary A/B before any adoption.
    /// Same clap shape as the tests above -- `num_args = 0..=1` and **no**
    /// `default_value` -- so the absent flag really is `None` and the
    /// `unwrap_or(false)` in `resolve` really is the decision.
    #[test]
    fn the_containment_clause_ships_off_and_both_arms_parse() {
        assert!(
            !cli(&[]).resolve(None).unwrap().defaults.containment_clause,
            "OFF by default -- a prompt change resamples everything"
        );
        assert!(
            cli(&["--containment-clause"])
                .resolve(None)
                .unwrap()
                .defaults
                .containment_clause,
            "the bare flag is the ON arm of the A/B"
        );
        assert!(
            !cli(&["--containment-clause", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .containment_clause,
            "an explicit false must stay expressible for the day the default flips"
        );
    }

    /// **The clipped-column repair ships ON, and both arms are pinned.**
    ///
    /// Same clap shape as the tests above -- `num_args = 0..=1` and **no**
    /// `default_value` -- so the `unwrap_or` really is the decision and a
    /// prove-red on it is a wired test rather than a blind one.
    #[test]
    fn the_clipped_column_repair_is_on_by_default_and_can_be_turned_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.repair_clipped_columns,
            "ships ON, after both arms rendered"
        );
        assert!(
            !cli(&["--repair-clipped-columns", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .repair_clipped_columns,
            "the off arm is what lowers the detector's TEXT request back to normal"
        );
    }

    /// **The axis tie-break ships ON, and both arms are pinned.**
    /// Adopted on the rendered two-corpus census;
    /// the ja/Unknown scope lives in `desired_config`, asserted there.
    ///
    /// Same clap shape as the flags above -- `num_args = 0..=1` and **no**
    /// `default_value` -- so the `unwrap_or(true)` really is the decision. A
    /// green after flipping that to `false` would mean a `default_value`
    /// sneaked onto the attribute and the resolve line is dead code.
    #[test]
    fn axis_aware_nms_ships_on_and_can_be_turned_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.axis_aware_nms,
            "ships ON, adopted on the census"
        );
        assert!(
            !cli(&["--axis-aware-nms", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .axis_aware_nms,
            "the off arm is the whole-pipeline kill switch the scope is not"
        );
    }

    /// Both device flags (`--nms-residue-regions`, `--strike-through-devices`)
    /// ship **ON**, on the whole-chapter render, which came out better on
    /// every page but three panels.
    ///
    /// They are asserted TOGETHER on purpose, because the render that earned
    /// the flip moved them together. No part of that verdict is attributable to
    /// either flag alone -- one page is the proof, with byte-identical regions and
    /// `rendered_text` yet 11,242 differing pixels, 10,096 of them the strike
    /// device on a page where the residue flag changed nothing. Reverting ONE
    /// of these alone no longer reproduces the arm that was evaluated.
    ///
    /// Same clap shape as the flags above -- `num_args = 0..=1` and **no**
    /// `default_value` -- so the `unwrap_or(true)` really is the decision.
    #[test]
    fn the_two_device_flags_ship_on_and_each_keeps_its_own_control_arm() {
        let shipped = cli(&[]).resolve(None).unwrap().defaults;
        assert!(
            shipped.nms_residue_regions,
            "ON by default, on the whole-chapter render"
        );
        assert!(
            shipped.strike_through_devices,
            "ON by default, on the same render"
        );

        // Each turns off ALONE, without dragging its neighbour -- which is what
        // makes a one-flag A/B possible at all. The two-flag arm above is
        // exactly the mistake this half of the test exists to keep available.
        let no_residue = cli(&["--nms-residue-regions", "false"])
            .resolve(None)
            .unwrap()
            .defaults;
        assert!(!no_residue.nms_residue_regions);
        assert!(
            no_residue.strike_through_devices,
            "the residue control arm must not silently also disarm the device"
        );

        let no_strike = cli(&["--strike-through-devices", "false"])
            .resolve(None)
            .unwrap()
            .defaults;
        assert!(!no_strike.strike_through_devices);
        assert!(
            no_strike.nms_residue_regions,
            "the device control arm must not silently also disarm the residue"
        );
    }

    /// The THIRD flag from the same work, `--segment-context`, was **not** part
    /// of that decision and still ships OFF. Flipping it along with the other
    /// two would ship an unrendered prompt change, and a prompt change re-rolls
    /// every page in the chapter -- so it cannot be reverted by looking at one
    /// panel.
    #[test]
    fn segment_context_is_not_one_of_the_flipped_device_flags() {
        assert!(
            !cli(&[]).resolve(None).unwrap().defaults.segment_context,
            "OFF -- a prompt change ships selectable, not on"
        );
        assert!(
            cli(&["--segment-context"]).resolve(None).unwrap().defaults.segment_context,
            "asking for it must still deliver it"
        );
    }

    /// The ink-colour styling's arm switch. Same clap shape as the flags
    /// above -- `num_args = 0..=1` and **no** `default_value` -- so the
    /// `unwrap_or(true)` really is the decision.
    #[test]
    fn sampled_ink_lettering_ships_on_and_can_be_turned_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.sampled_ink_lettering,
            "ships ON -- the styling's shipping arm"
        );
        assert!(
            !cli(&["--sampled-ink-lettering", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .sampled_ink_lettering,
            "false is the same-binary control arm the A/B renders on"
        );
    }

    /// **ON by default, on the rendered A/Bs** (a Korean test chapter: one
    /// fire in 67 slices -- the exhibit -- with 143/144 byte-identical; a
    /// Chinese control chapter: zero fires, 340/340 byte-identical).
    #[test]
    fn replace_scream_marks_ships_on_and_false_is_the_control_arm() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.replace_scream_marks,
            "ON by default, on the rendered A/Bs"
        );
        assert!(
            !cli(&["--replace-scream-marks", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .replace_scream_marks,
            "false is the same-binary control arm"
        );
        let defaults = cli(&[]).resolve(None).unwrap().defaults;
        assert!(
            defaults.spot_rescue_erase,
            "the device rides the erase flag's shipping ON -- inert without it"
        );
    }

    /// **The joined-page floor ships at 0.20, and `0` disables it.** Adopted
    /// with the tie-break. A floor of `0` admitting every
    /// detection the network ever returned would be the worst possible
    /// fat-finger; mirroring `--large-crop-ocr 0` makes it the off switch
    /// instead.
    #[test]
    fn the_joined_page_floor_ships_at_the_censused_value_and_zero_disables_it() {
        assert_eq!(
            cli(&[]).resolve(None).unwrap().defaults.joined_page_text_floor,
            Some(0.20),
            "ships at the censused 0.20"
        );
        assert_eq!(
            cli(&["--joined-page-text-floor", "0.15"])
                .resolve(None)
                .unwrap()
                .defaults
                .joined_page_text_floor,
            Some(0.15),
            "an explicit value wins over the adopted default"
        );
        assert_eq!(
            cli(&["--joined-page-text-floor", "0"])
                .resolve(None)
                .unwrap()
                .defaults
                .joined_page_text_floor,
            None,
            "zero is the off switch, not a floor that admits everything"
        );
    }

    /// **The turned second read ships ON, and both arms are pinned.**
    #[test]
    fn the_rotated_reread_is_on_by_default_and_can_be_turned_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.reread_rotated_columns,
            "ships ON, on the renders"
        );
        assert!(
            !cli(&["--reread-rotated-columns", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .reread_rotated_columns,
            "the off arm is the only way to see what the unturned read returned"
        );
    }

    /// **The upright pass ships ON, and both arms are pinned.**
    /// Flipped after the chapter render was looked at; the OFF arm is pinned so
    /// a control render can still be taken against the pre-pass behaviour.
    #[test]
    fn the_upright_pass_is_on_by_default_and_can_be_turned_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.upright_pass,
            "ON by default, on the render"
        );
        assert!(
            !cli(&["--upright-pass", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .upright_pass,
            "the off arm is the byte-identical pre-pass control"
        );
        assert!(
            cli(&["--upright-pass"])
                .resolve(None)
                .unwrap()
                .defaults
                .upright_pass,
            "the bare flag still means ON, same as before the default moved"
        );
    }

    /// **The flip re-read ships ON, and both arms are pinned.**
    /// Flipped on its render; the OFF arm is pinned so a control render
    /// can still be taken against the pre-flip behaviour -- the same road
    /// `--upright-pass` and `--read-textless-bubbles` travelled.
    #[test]
    fn the_flip_reread_is_on_by_default_and_can_be_turned_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.flip_reread_bubbles,
            "ON by default, on the render"
        );
        assert!(
            !cli(&["--flip-reread-bubbles", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .flip_reread_bubbles,
            "the off arm is the byte-identical pre-flip control"
        );
        assert!(
            cli(&["--flip-reread-bubbles"])
                .resolve(None)
                .unwrap()
                .defaults
                .flip_reread_bubbles,
            "the bare flag still means ON, same as before the default moved"
        );
    }

    /// **The spot rescue ships ON, both flags, and all four arms are pinned.**
    /// The lettering and the erase are separate flags so their
    /// renders can adjudicate them separately -- the erase's risk is a
    /// big-box erase of artwork, the lettering's is a wrong skill name, and
    /// neither goes on without its own render.
    #[test]
    fn the_spot_rescue_ships_on_and_the_erase_stays_its_own_flag() {
        // ON by default, on the rendered chapter checked against the official
        // edition (zero 500s, the rescued display texts adjudicated, 328/335
        // images byte-identical to the OFF arm). The ERASE keeps its own
        // default and its own render.
        let defaults = cli(&[]).resolve(None).unwrap().defaults;
        assert!(defaults.spot_rescue, "ON by default, on its render");
        assert!(
            defaults.spot_rescue_erase,
            "ON by default, on its OWN render: changed exactly two images of \
             335, both better, 780 s vs 775 s"
        );
        assert!(
            !cli(&["--spot-rescue", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .spot_rescue,
            "an explicit false is the control arm"
        );
        let no_erase = cli(&["--spot-rescue-erase", "false"])
            .resolve(None)
            .unwrap()
            .defaults;
        assert!(
            no_erase.spot_rescue && !no_erase.spot_rescue_erase,
            "the erase's control arm must not drag the lettering off"
        );
        let no_rescue = cli(&["--spot-rescue", "false"]).resolve(None).unwrap().defaults;
        assert!(
            no_rescue.spot_rescue_erase,
            "the flags stay independent in the other direction too; with the \
             rescue off the erase value is moot but must not be rewritten"
        );
    }

    /// **The joined trigger ships ON, on the rendered chapter** (2 of 179
    /// reader-final slices change, the joined column delivers, 177/179
    /// byte-identical as the A/A).
    #[test]
    fn the_joined_spot_trigger_is_on_by_default_and_false_is_the_control_arm() {
        let defaults = cli(&[]).resolve(None).unwrap().defaults;
        assert!(
            defaults.spot_rescue_joined,
            "ON by default, on its own render"
        );
        assert!(
            cli(&["--spot-rescue-joined"])
                .resolve(None)
                .unwrap()
                .defaults
                .spot_rescue_joined,
            "the bare flag still reads as true"
        );
        assert!(
            !cli(&["--spot-rescue-joined", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .spot_rescue_joined,
            "an explicit false is the control arm"
        );
    }

    /// **The orientation ranking ships at `0.17`, and the off arm is `inf`.**
    /// `0.17` is the value the chapter was rendered at (3 of 179 pages change,
    /// rescue population byte-identical). `None` -- "do not rank" -- is no
    /// longer reachable from the CLI; the off arm is an infinite margin, which
    /// no finite confidence gap can clear. `0.0` is deliberately NOT off (it
    /// flips ties); `choose_orientation`'s own tests pin both facts.
    #[test]
    fn the_orientation_margin_ships_at_0_17_and_inf_is_the_off_arm() {
        assert_eq!(
            cli(&[])
                .resolve(None)
                .unwrap()
                .defaults
                .orientation_confidence_margin,
            Some(0.17),
            "the shipping default is the rendered value, not an untested one"
        );
        assert_eq!(
            cli(&["--orientation-confidence-margin", "0.30"])
                .resolve(None)
                .unwrap()
                .defaults
                .orientation_confidence_margin,
            Some(0.30),
            "an explicit value must override the default, not merge with it"
        );
        let off = cli(&["--orientation-confidence-margin", "inf"])
            .resolve(None)
            .unwrap()
            .defaults
            .orientation_confidence_margin;
        assert_eq!(
            off,
            Some(f64::INFINITY),
            "`inf` must parse: it is the documented off arm for this flag"
        );
    }

    /// **The unjoined-page column turn ships ON, and both arms are pinned.**
    #[test]
    fn the_unjoined_column_turn_is_on_by_default_and_can_be_turned_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.turn_unjoined_columns,
            "ships ON, on the rendered cost"
        );
        assert!(
            !cli(&["--turn-unjoined-columns", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .turn_unjoined_columns,
            "the off arm restores `rotate_free_text_columns && joined_page`"
        );
    }

    /// **The scoped watermark refusal ships ON in BOTH of its copies.**
    ///
    /// This is the sharpest of the default pins. The default is written **twice** --
    /// `defaults.scope_watermark_refusals` for the letter side and
    /// `Resolved::scope_watermark_refusals` for the erase side -- and the comment
    /// beside the second says *"this `unwrap_or` must be flipped with the other
    /// one"*. Nothing enforced that. A test asserting only one copy would pass with
    /// the two silently disagreeing -- a test of two halves does not test the
    /// `||` between them: assert what the callers actually read, and there are
    /// two of them.
    #[test]
    fn the_scoped_watermark_refusal_is_on_in_both_copies_and_can_be_turned_off() {
        let on = cli(&[]).resolve(None).unwrap();
        assert!(on.defaults.scope_watermark_refusals, "letter side ships ON");
        assert!(on.scope_watermark_refusals, "erase side ships ON");
        assert_eq!(
            on.defaults.scope_watermark_refusals, on.scope_watermark_refusals,
            "the two copies must agree; they are read from the same Option and \
             cli.rs's own comment says they must be flipped together"
        );

        let off = cli(&["--scope-watermark-refusals", "false"]).resolve(None).unwrap();
        assert!(!off.defaults.scope_watermark_refusals, "letter side turns off");
        assert!(!off.scope_watermark_refusals, "erase side turns off");
        assert_eq!(
            off.defaults.scope_watermark_refusals, off.scope_watermark_refusals,
            "and they must still agree in the off arm, which is the arm a divergence \
             would hide in"
        );
    }

    /// The two misread levers, same two-copies contract as the watermark scope
    /// above: each flag is read into a letter-side copy (`Resolved`, through
    /// `Shared`) and an erase-side copy (`defaults`, through `ProcessorConfig`),
    /// and a divergence is a region refused by one gate and erased by the other.
    #[test]
    fn the_misread_levers_ship_on_in_both_copies_and_flip_together() {
        // ON, on a read of the censused Korean chapters (zh control
        // byte-identical 340/340).
        let on = cli(&[]).resolve(None).unwrap();
        assert!(on.defaults.leave_misread_bubbles, "erase side ships ON");
        assert!(on.leave_misread_bubbles, "letter side ships ON");
        assert!(on.defaults.korean_script_strict, "erase side ships ON");
        assert!(on.korean_script_strict, "letter side ships ON");
        assert_eq!(on.defaults.leave_misread_bubbles, on.leave_misread_bubbles);
        assert_eq!(on.defaults.korean_script_strict, on.korean_script_strict);

        // The off arm is the census control, and both copies must turn together
        // there too -- the off arm is the arm a divergence would hide in.
        let off = cli(&["--leave-misread-bubbles", "false", "--korean-script-strict", "false"])
            .resolve(None)
            .unwrap();
        assert!(!off.defaults.leave_misread_bubbles && !off.leave_misread_bubbles);
        assert!(!off.defaults.korean_script_strict && !off.korean_script_strict);
    }

    /// The reserve re-read ships ON -- on the rendered evidence (one page
    /// changed, zero re-admissions) -- and
    /// the flag turns it off. Asserted on the field `desired_config` actually
    /// reads (`defaults.reread_refused_dialogue`), in both arms, so the
    /// `unwrap_or(true)` that IS the default is the line under test.
    #[test]
    fn the_reserve_reread_ships_on_and_the_flag_turns_it_off() {
        let on = cli(&[]).resolve(None).unwrap();
        assert!(
            on.defaults.reread_refused_dialogue,
            "ON by default, on the rendered evidence"
        );
        let bare = cli(&["--reread-refused-dialogue"]).resolve(None).unwrap();
        assert!(bare.defaults.reread_refused_dialogue, "the bare flag arms it");
        let off = cli(&["--reread-refused-dialogue", "false"])
            .resolve(None)
            .unwrap();
        assert!(
            !off.defaults.reread_refused_dialogue,
            "false is the census off arm"
        );
    }

    /// **The implausible-text gate ships ON, and it is spelled as an inversion.**
    ///
    /// Without this test it would have the identical failure shape: zero
    /// coverage in either arm. `skip_implausible_text` is `!letter_implausible_text`,
    /// so the flag a reader passes and the field the pipeline reads have opposite
    /// senses; the test asserts the field the caller actually reads, in both arms,
    /// rather than the flag it was spelled from.
    #[test]
    fn the_implausible_text_gate_is_on_by_default_and_can_be_turned_off() {
        assert!(
            cli(&[]).resolve(None).unwrap().skip_implausible_text,
            "the gate ships ON; the flag that turns it off is the positive-sounding one"
        );
        assert!(
            !cli(&["--letter-implausible-text"])
                .resolve(None)
                .unwrap()
                .skip_implausible_text,
            "passing --letter-implausible-text turns the SKIP off, not on"
        );
    }

    /// The source-ink fraction ships at its measured 0.12 and is overridable.
    ///
    /// Asserted against `lettering::SOURCE_INK_FRACTION` rather than a literal,
    /// so the flag and the constant cannot drift apart -- the whole point of
    /// `default_value_t` here is that there is one number, not two. The override
    /// arm is what makes an A/B of the fraction possible at all.
    #[test]
    fn the_source_ink_fraction_defaults_to_the_measured_constant() {
        assert_eq!(
            cli(&[]).resolve(None).unwrap().source_ink_fraction,
            crate::lettering::SOURCE_INK_FRACTION
        );
        assert_eq!(
            cli(&["--source-ink-fraction", "0.30"])
                .resolve(None)
                .unwrap()
                .source_ink_fraction,
            0.30
        );
    }

    /// Oversized-crop routing is ON at 448 by default, flipped on a real-page
    /// regression check: 3 of 453 regions route, 37 of 40 pages stay
    /// byte-identical, and two of the three were catastrophic manga-ocr
    /// failures. Zero regressions.
    #[test]
    fn large_crop_routing_is_on_by_default() {
        assert_eq!(
            cli(&[]).resolve(None).unwrap().defaults.large_crop_ocr_px,
            Some(448)
        );
        assert_eq!(
            cli(&["--large-crop-ocr", "672"])
                .resolve(None)
                .unwrap()
                .defaults
                .large_crop_ocr_px,
            Some(672)
        );
    }

    /// Zero has to mean "off" rather than "route everything". It is the natural
    /// thing a script passes to disable a numeric flag, and now that the flag
    /// defaults to 448 it is also the ONLY way back to the pre-routing behaviour.
    #[test]
    fn a_zero_threshold_disables_routing_rather_than_routing_everything() {
        assert_eq!(
            cli(&["--large-crop-ocr", "0"])
                .resolve(None)
                .unwrap()
                .defaults
                .large_crop_ocr_px,
            None
        );
    }

    /// The ceiling is ON at 0.5. It sits in a gap with nothing measured in it:
    /// the largest legitimate routed region on the run that latched is 0.30x of
    /// the page, the mis-segmented box that latched it is 1.49x, and the largest
    /// region the large-crop routing was ever measured on is 0.16x.
    #[test]
    fn the_region_area_ceiling_is_on_by_default() {
        let resolved = cli(&[]).resolve(None).unwrap();
        assert_eq!(resolved.defaults.large_crop_ocr_max_area, Some(0.5));
        assert_eq!(
            cli(&["--large-crop-ocr-max-area", "1.0"])
                .resolve(None)
                .unwrap()
                .defaults
                .large_crop_ocr_max_area,
            Some(1.0)
        );
    }

    /// Zero has to mean "off" here for the same reason it does on
    /// `--large-crop-ocr`: it is the only way back to the pre-ceiling behaviour,
    /// and it is what a measurement script passes for the other arm.
    #[test]
    fn a_zero_area_ceiling_disables_it_rather_than_refusing_everything() {
        assert_eq!(
            cli(&["--large-crop-ocr-max-area", "0"])
                .resolve(None)
                .unwrap()
                .defaults
                .large_crop_ocr_max_area,
            None
        );
    }

    /// A negative fraction would otherwise be filtered into `None` and read as
    /// the ceiling working while it is not there at all. `0` is the documented
    /// off switch; a second, silent spelling of it is what this refuses.
    ///
    /// NaN is the one that has to be caught rather than merely reported:
    /// `PipelineConfig` derives `PartialEq` and `needs_reload` compares it, so a
    /// NaN in `processor` makes every request unequal to the applied config and
    /// reloads the pipeline -- discarding the residency profiles -- on every
    /// single page.
    ///
    /// Written with `=` because clap reads a bare `-0.5` as a flag.
    #[test]
    fn a_negative_or_nan_area_ceiling_is_refused_rather_than_silently_turned_off() {
        for argument in ["--large-crop-ocr-max-area=-0.5", "--large-crop-ocr-max-area=nan"] {
            // `.err().expect(..)` rather than `unwrap_err()`: the latter needs
            // `Resolved: Debug` to format the Ok arm it is asserting cannot happen,
            // and nothing else in this crate requires that derive.
            let error = cli(&[argument])
                .resolve(None)
                .err()
                .expect("a negative or NaN ceiling must be refused")
                .to_string();
            assert!(error.contains("--large-crop-ocr-max-area"), "{error}");
            assert!(error.contains("0 to turn the ceiling off"), "{error}");
        }
    }

    /// A region the area ceiling refuses is dropped from OCR entirely, because
    /// handing it to the primary is what put a fabricated `AND SO,` across a
    /// textless panel of the measured volume. Both arms are pinned so the
    /// default cannot drift back silently.
    #[test]
    fn an_implausible_region_is_dropped_rather_than_read_by_the_primary() {
        assert!(
            cli(&[])
                .resolve(None)
                .unwrap()
                .defaults
                .skip_implausible_regions
        );
        assert!(
            !cli(&["--skip-implausible-regions", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .skip_implausible_regions
        );
    }

    /// The erase-mask arm ships ON, and the pixels are why.
    ///
    /// The bar was that it goes on only once a comparison on the mis-segmented
    /// page shows the artwork preserved with nothing else regressing. Run with
    /// `clean_only` over 20 consecutive pages, both arms on one binary:
    /// **19 of 20 pages byte-identical, the mis-segmented page changes 17.6% of
    /// its pixels, and the change is the whole panel coming back.** Off, LaMa
    /// smears a character's whole figure into blank white; on, the page is
    /// indistinguishable from the source. Collateral over the 20 pages falls
    /// 2.62% -> 1.74%.
    ///
    /// The assertion is now the other way round, and it guards the same thing it
    /// always did: a default that drifted would ship an unmeasured change to
    /// what gets erased.
    #[test]
    fn the_erase_mask_ceiling_is_on_because_the_pixels_said_so() {
        assert!(
            cli(&[])
                .resolve(None)
                .unwrap()
                .defaults
                .skip_implausible_masks
        );
        assert!(
            !cli(&["--skip-implausible-masks", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .skip_implausible_masks
        );
        // The two decisions are separate flags and must stay independently
        // reachable: refusing to READ the box and refusing to ERASE it are
        // different measurements, and each has its own arm.
        let no_erase_gate = cli(&["--skip-implausible-masks", "false"])
            .resolve(None)
            .unwrap();
        assert!(no_erase_gate.defaults.skip_implausible_regions);
        let no_read_gate = cli(&["--skip-implausible-regions", "false"])
            .resolve(None)
            .unwrap();
        assert!(no_read_gate.defaults.skip_implausible_masks);
    }

    /// The unread-mask veto ships **ON**, and turning off its sibling must not turn
    /// it off too.
    ///
    /// **The coupling is the defect this flag exists to remove**, so a test that
    /// only checked the default would miss the whole point. With
    /// `--skip-implausible-regions` ON, a size-refused box never becomes an OCR
    /// result, and `--withdraw-illegible-masks` only walks results — so one flag
    /// silently disabled the other on 22 boxes and 18 of them were erased.
    ///
    /// **This test once asserted the opposite, deliberately.** It shipped OFF
    /// until the render and the panel landed, and was then flipped.
    /// The test survives the flip with its polarity reversed, which is what it was
    /// written for — the next move of this default also has to be deliberate.
    #[test]
    fn the_unread_mask_veto_is_on_by_default_and_is_independent_of_its_sibling() {
        assert!(
            cli(&[]).resolve(None).unwrap().defaults.withdraw_unread_masks,
            "shipping default must withdraw a region no engine was allowed to read"
        );
        assert!(
            !cli(&["--withdraw-unread-masks", "false"])
                .resolve(None)
                .unwrap()
                .defaults
                .withdraw_unread_masks,
            "the A/B escape hatch must reach the old behaviour"
        );

        // Both directions, because "independent" is a claim about the PAIR and
        // asserting each flag alone would pass with the two wired together --
        // which is exactly the state this flag was written to fix.
        //
        // Turning the sibling off and naming NOTHING else, so the assertion is
        // about the DEFAULT rather than about an explicit override. Passing
        // `--withdraw-unread-masks true` here would pass even if the default were
        // coupled to the sibling, which is the failure this flag exists to fix.
        let no_illegible = cli(&["--withdraw-illegible-masks", "false"])
            .resolve(None)
            .unwrap();
        assert!(!no_illegible.defaults.withdraw_illegible_masks);
        assert!(
            no_illegible.defaults.withdraw_unread_masks,
            "turning the illegible veto off must not take the unread veto with it"
        );

        let no_unread = cli(&["--withdraw-unread-masks", "false"]).resolve(None).unwrap();
        assert!(no_unread.defaults.withdraw_illegible_masks);
        assert!(!no_unread.defaults.withdraw_unread_masks);

        // And it is orthogonal to the reader gate it depends on for its input:
        // the flag is settable whether or not anything is being refused for size.
        let no_read_gate = cli(&["--skip-implausible-regions", "false", "--withdraw-unread-masks", "true"])
            .resolve(None)
            .unwrap();
        assert!(!no_read_gate.defaults.skip_implausible_regions);
        assert!(no_read_gate.defaults.withdraw_unread_masks);
    }

    #[test]
    fn watch_pid_is_off_by_default_and_reaches_resolved_when_named() {
        // Off means no watch at all -- a bare server run (harness scripts,
        // tests) must not inherit a watchdog on some unrelated process.
        assert_eq!(cli(&[]).resolve(None).unwrap().watch_pid, None);
        assert_eq!(
            cli(&["--watch-pid", "4242"]).resolve(None).unwrap().watch_pid,
            Some(4242),
            "serve.ps1 passes its own PID and it must survive resolve()"
        );
    }

    #[test]
    fn an_explicit_font_family_replaces_the_default_rather_than_extending_it() {
        // The escape hatch from the bundled catalog is naming a system family,
        // so it has to actually drop CCWildWords -- appending would leave it
        // first in the list and change nothing.
        let resolved = cli(&["--font-family", "Arial"]).resolve(None).unwrap();
        assert_eq!(resolved.font_families, ["Arial"]);
    }

    #[test]
    fn a_zero_idle_budget_disables_the_unload_rather_than_firing_at_once() {
        assert_eq!(
            cli(&["--idle-unload-secs", "0"]).resolve(None).unwrap().idle_unload,
            None
        );
        assert_eq!(
            cli(&["--idle-unload-secs", "45"])
                .resolve(None)
                .unwrap()
                .idle_unload,
            Some(Duration::from_secs(45))
        );
    }

    #[test]
    fn the_cold_estimate_follows_the_provider_unless_it_is_given() {
        // The local engine holds the translation model in this process, so it
        // has to be budgeted for; under ollama those weights are elsewhere.
        assert_eq!(
            cli(&[]).resolve(None).unwrap().cold_reserve_bytes,
            vram::COLD_RESERVE_STAGES + vram::COLD_RESERVE_LOCAL_LLM
        );
        assert_eq!(
            cli(&["--provider", "ollama", "--llm", "qwen3:8b"])
                .resolve(None)
                .unwrap()
                .cold_reserve_bytes,
            vram::COLD_RESERVE_STAGES
        );
        assert_eq!(
            cli(&["--cold-reserve-bytes", "123"])
                .resolve(None)
                .unwrap()
                .cold_reserve_bytes,
            123
        );
    }

    #[test]
    fn the_endpoint_defaults_to_a_local_ollama() {
        let resolved = cli(&[]).resolve(None).unwrap();
        assert_eq!(
            resolved
                .providers
                .openai_compatible
                .base_url
                .unwrap()
                .as_str(),
            "http://localhost:11434/v1"
        );
    }

    #[test]
    fn a_base_url_override_is_honoured() {
        let resolved = cli(&[
            "--provider",
            "ollama",
            "--llm",
            "qwen3:8b",
            "--base-url",
            "http://192.0.2.50:11434/v1",
        ])
        .resolve(None)
        .unwrap();
        assert_eq!(
            resolved
                .providers
                .openai_compatible
                .base_url
                .unwrap()
                .as_str(),
            "http://192.0.2.50:11434/v1"
        );
    }

    #[test]
    fn hunyuan_substitute_replaces_the_hunyuan_default_and_nothing_else() {
        use crate::models::ocr_model_name;
        use koharu_pipeline::OcrModel;

        let on = cli(&["--hunyuan-substitute", "paddleocr-vl-1.6"]).resolve(None).unwrap();
        assert_eq!(ocr_model_name(&on.defaults.ocr), "paddleocr-vl-1.6");
        assert_eq!(on.defaults.ocr_substitute, Some(OcrModel::PaddleOcrVl1_6));

        let other = cli(&["--ocr", "manga-ocr", "--hunyuan-substitute", "paddleocr-vl-1.6"])
            .resolve(None)
            .unwrap();
        assert_eq!(ocr_model_name(&other.defaults.ocr), "manga-ocr");

        let off = cli(&[]).resolve(None).unwrap();
        assert_eq!(ocr_model_name(&off.defaults.ocr), "hunyuan-ocr-1.5");
        assert_eq!(off.defaults.ocr_substitute, None);
    }

    #[test]
    fn startup_refuses_a_hunyuan_or_unknown_substitute() {
        let Err(error) = cli(&["--hunyuan-substitute", "hunyuan-ocr-1.5"]).resolve(None) else {
            panic!("hunyuan-ocr-1.5 was accepted as its own substitute");
        };
        assert!(error.to_string().contains("cannot be hunyuan-ocr-1.5"), "{error}");
        let Err(error) = cli(&["--hunyuan-substitute", "no-such-engine"]).resolve(None) else {
            panic!("an unknown engine was accepted");
        };
        // Names the flag, and never offers the one value it refuses.
        let error = error.to_string();
        assert!(error.starts_with("bad --hunyuan-substitute \"no-such-engine\""), "{error}");
        assert!(!error.contains("hunyuan-ocr-1.5"), "{error}");
        // Every engine it does offer is one it accepts.
        let offered = error.split_once("; use ").unwrap().1;
        for engine in offered.split(", ") {
            assert!(cli(&["--hunyuan-substitute", engine]).resolve(None).is_ok(), "{engine}");
        }
    }

    #[test]
    fn store_dir_flag_wins_over_the_environment_and_is_made_absolute() {
        let env = || Some(std::ffi::OsString::from("from-env"));

        let flag = cli(&["--store-dir", "from-flag"]).resolve_store_dir(env()).unwrap().unwrap();
        assert!(flag.is_absolute() && flag.ends_with("from-flag"), "{}", flag.display());

        let from_env = cli(&[]).resolve_store_dir(env()).unwrap().unwrap();
        assert!(from_env.is_absolute() && from_env.ends_with("from-env"), "{}", from_env.display());

        // Unset and empty both leave Koharu's default alone.
        assert_eq!(cli(&[]).resolve_store_dir(None).unwrap(), None);
        assert_eq!(cli(&[]).resolve_store_dir(Some("".into())).unwrap(), None);

        let absolute = std::env::temp_dir().join("birelate-store");
        let given = absolute.to_str().unwrap();
        assert_eq!(
            cli(&["--store-dir", given]).resolve_store_dir(env()).unwrap(),
            Some(absolute)
        );
    }
}
