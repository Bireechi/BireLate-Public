//! Server state, and the rules governing what a single request may change.
//!
//! Three classes of setting, decided by how each one behaves in VRAM:
//!
//! * The OpenAI-compatible endpoint is fixed at startup. It could be per
//!   request -- the translator re-reads it on every call -- but the extension
//!   has no field for it.
//! * OCR, inpainting, target language and instructions are per request. They
//!   are snapshotted into a stage runner at construction, so changing them
//!   means rebuilding that runner, which allocates nothing until the next run
//!   loads a model.
//! * The provider is pinned for the life of the process. Switching the local
//!   model loads the new weights before dropping the old, and the residency
//!   guard cannot evict the old ones because it only considers stages that
//!   report themselves loaded -- which is false precisely when the selection
//!   changed. On a 32 GB card that is a native abort with no diagnostics, so
//!   the server refuses instead.
//! * The LLM name follows the provider. Under `ollama` it is per request: the
//!   weights live in Ollama's process, so naming a different tag costs this
//!   address space nothing. Under `local` it is pinned, for the reason above.

use std::num::NonZeroU32;
use std::{
    panic::AssertUnwindSafe,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use axum::http::StatusCode;
use futures::FutureExt as _;
use koharu_pipeline::{
    InpaintingModel, OcrModel, Pipeline, PipelineConfig, PipelineError, ResourceSnapshot, Stage,
    TranslationConfig,
};
use koharu_translator::{GenerationConfig, Language, ModelSelection, Provider};

use crate::{
    error::{ApiError, CLIP_BUDGET, clip, panic_text, root_cause},
    guard::GuardState,
    idle::IdleClock,
    models,
    render::Renderers,
};

/// Paid only when a request genuinely changes a model. Dropping a stage runner
/// releases its weights, but the residency logic elsewhere waits after an
/// unload too, so the release is evidently not instantaneous.
pub const SETTLE_AFTER_RELOAD: Duration = Duration::from_millis(250);

/// How much of a pinned name is echoed in the 409, so the message stays inside
/// the extension's window.
const ECHO: usize = 24;

#[derive(Clone)]
pub struct AppState(pub Arc<Shared>);

impl std::ops::Deref for AppState {
    type Target = Shared;

    fn deref(&self) -> &Shared {
        &self.0
    }
}

pub struct Shared {
    /// The pipeline, built once and never dropped. Its config watcher owns the
    /// sender that keeps itself alive, so the task is immortal: dropping the
    /// pipeline would leak the active stage runner, its loaded weights and the
    /// translator's local model for the rest of the process.
    pub pipeline: Pipeline,
    pub renderers: Arc<Renderers>,
    /// Everything that touches the device happens under this, as one unit:
    /// reconcile, execute, render. The pipeline's own lock covers only execute,
    /// which is not enough -- a reconcile landing between ours and our execute
    /// would run someone else's models.
    pub gate: Arc<tokio::sync::Mutex<GpuState>>,
    pub defaults: Defaults,
    pub pinned: Pinned,
    pub guard: GuardState,
    /// Empty by default, on purpose: see `render::render_request`.
    pub font_families: Vec<String>,
    /// Page-wide hyphenation override, or `None` for the renderer's own rule.
    pub hyphenation: Option<koharu_renderer::HyphenationPolicy>,
    /// How far above the page's prevailing dialogue size one balloon may be set.
    /// `None` leaves every balloon solving alone, which is the A/B's other arm.
    pub size_coherence: Option<f32>,
    pub collision_relief: bool,
    /// Whether a cut balloon's lettering anchors at the source ink's center.
    /// On by default -- see `no_edge_anchored_lettering` in
    /// `cli.rs`.
    pub edge_anchored_lettering: bool,
    /// Letter translated dialogue in capitals. Off by default: it pairs with a
    /// comic face, and the fallback here is Arial.
    pub uppercase_dialogue: bool,
    /// Whether `labels::hide_implausible` runs. On by default -- see the flag's
    /// doc in `cli.rs` for the 511-segment validation and the pixel case.
    pub skip_implausible_text: bool,
    /// Whether that refusal is scoped to the site's own text rather than
    /// condemning the whole region. **ON by default.**
    pub scope_watermark_refusals: bool,
    /// Whether a script/punctuation refusal reaches DIALOGUE-role regions on a
    /// declared zh/ko page -- the lettering half of a paired lever. The mask
    /// half is `ProcessorConfig::leave_misread_bubbles`, written from the same
    /// flag in `resolve` so the two cannot disagree. **ON by default.**
    pub leave_misread_bubbles: bool,
    /// Whether a declared-Korean read with no hangul is a script mismatch --
    /// `labels::strict_korean_mismatch`. Same paired-write contract as above.
    /// **ON by default**, with its pair.
    pub korean_script_strict: bool,
    pub skip_duplicate_text: bool,
    /// Whether the duplicate gate uses the drawn frame instead of its AABB.
    /// **On by default**; see `no_duplicate_oriented_overlap`'s
    /// doc in `cli.rs` for the corpus numbers that moved it.
    pub duplicate_oriented_overlap: bool,
    /// Whether that gate additionally requires the two SOURCE boxes to
    /// coincide before a pair may drop. **On by default**;
    /// `--duplicate-shared-source` in `cli.rs` has the numbers.
    pub duplicate_shared_source: bool,
    /// Whether the free-standing size cap runs. Off makes the A/B possible.
    pub fit_free_text: bool,
    /// The source-ink fraction that cap solves for. See `--source-ink-fraction`;
    /// it is the term that decides how large a vertical caption may be lettered.
    pub source_ink_fraction: f32,
    /// Pinned English for drawn sound effects. Empty unless `--sfx-dictionary`
    /// was given, and inert when empty. On `Shared` rather than `Defaults`
    /// because it is applied to the finished scene, not baked into
    /// `PipelineConfig`, so changing it would never need a pipeline reload.
    pub sfx_dictionary: crate::sfx::Dictionary,
    pub max_upload_bytes: usize,
    /// How long a request may wait for the gate. Never a cap on a running job:
    /// the extension fires one un-awaited request per image on a page, so a
    /// gallery legitimately queues several minutes of work.
    pub queue_timeout: Duration,
    /// The pipeline's own VRAM telemetry, subscribed once at startup.
    ///
    /// Held rather than re-subscribed per request for two reasons: the monitor
    /// only starts sampling from inside a Tokio runtime, and reading a watch
    /// value is a lock-free borrow, so a status request costs nothing and
    /// touches no device.
    pub vram: tokio::sync::watch::Receiver<ResourceSnapshot>,
    pub idle: IdleClock,
    /// Whether a warmup is already running, so a second `/warmup` is a no-op
    /// rather than a second set of model loads racing the first. Take it through
    /// `claim_warmup`, which is the only correct way to read it.
    pub warming: AtomicBool,
    /// What one cold page is assumed to need. An estimate; see `crate::vram`.
    pub cold_reserve_bytes: u64,
    /// Fires the graceful shutdown, for the popup's Stop button.
    ///
    /// A `Notify` rather than a oneshot because `Shared` is behind an `Arc` and
    /// handlers only ever hold `&self`: consuming a sender would need interior
    /// mutability and a "already fired" state, where `notify_waiters` is
    /// idempotent by construction. Two clicks are one shutdown.
    pub shutdown: Arc<tokio::sync::Notify>,
    /// Per-story translation context. In memory only, so stopping the server
    /// forgets every story -- which is the same boundary the reader's Stop
    /// button draws for the image cache.
    pub stories: Arc<crate::story::Stories>,
    /// Whether an `onomatopoeia`-labelled region's pair stays out of the story
    /// window -- `--story-excludes-sfx`, read at the one record site in
    /// `routes.rs` through `regions::feeds_story`.
    pub story_excludes_sfx: bool,
    /// Per-series pinned terms, keyed by the same story id. A sibling of
    /// `stories`, not a tenant: "New story" clears context and leaves the
    /// glossary standing -- the series' terms outlive a reading session.
    pub glossaries: Arc<crate::glossary::Glossaries>,
}

impl Shared {
    /// Takes the right to start a warmup, or reports that someone already holds
    /// it. Released by `WarmupInFlight` however the load ends.
    pub fn claim_warmup(&self) -> bool {
        self.warming
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

/// Lives inside the gate, so it needs no lock of its own.
pub struct GpuState {
    /// What is actually installed in the pipeline right now.
    pub applied: PipelineConfig,
}

#[derive(Clone, Debug)]
pub struct Defaults {
    /// Already substituted: see `ocr_substitute`.
    pub ocr: OcrModel,
    /// `--hunyuan-substitute`: the engine that serves every request for
    /// `hunyuan-ocr-1.5`, or `None` to serve Hunyuan itself. Applied by
    /// `models::resolve_ocr` and reported by `/status`.
    pub ocr_substitute: Option<OcrModel>,
    pub inpainting: InpaintingModel,
    pub target_language: Language,
    pub instructions: Option<String>,
    /// Whether the translator prompt carries the containment sentence.
    /// **OFF by default**; see `Cli::containment_clause`.
    pub containment_clause: bool,
    /// Whether each segment is described to the translator -- its kind, and
    /// whether the artwork strikes it through. **OFF by default**; see
    /// `Cli::segment_context`. The reader may override it per request
    /// through the popup toggle; this is the fallback when they do not.
    pub segment_context: bool,
    /// `None` leaves the model descriptor's own sampling alone. See
    /// `Cli::translation_temperature`.
    pub temperature: Option<f32>,
    /// `None` leaves llama.cpp's own KV-cache sizing alone. See `Cli::swa_full`
    /// for the arithmetic and for why the reduced cache is sound on this
    /// pipeline. Fixed for the process: it is part of `PipelineConfig`, so
    /// changing it per request would force a reload and discard the residency
    /// profiles.
    pub swa_full: Option<bool>,
    /// Whether RF-DETR's `onomatopoeia` class is lettered rather than erased.
    ///
    /// Lives here rather than on `Shared` because it is a *detection* setting
    /// that `desired_config` bakes into `PipelineConfig`, not a render override
    /// applied per page. It is fixed for the process: changing it would need a
    /// pipeline reload, and a reload discards the residency profiles.
    pub translate_sfx: bool,
    /// Whether the erase mask is sharpened by the text segmenter before it
    /// reaches the inpainter. Off by default; see `Cli::refine_text_mask`.
    pub refine_text_mask: bool,
    /// Denoising steps for RORem, or `None` for its own default. Process-wide,
    /// because it is part of the config a reload is decided on.
    pub rorem_steps: Option<i32>,
    /// FLUX replacement strength, or `None` for the configured default. Same
    /// reload rule as `rorem_steps`.
    pub flux_strength: Option<f64>,
    /// Confidence a sound effect needs, or `None` for the checkpoint's 0.2.
    pub onomatopoeia_threshold: Option<f32>,
    /// Per-region erase-mask scale, or `None` for the page-flat dilation.
    pub mask_scale: Option<f32>,
    /// Route a text region whose larger side reaches this many pixels to
    /// PaddleOCR-VL. Part of `PipelineConfig`, so changing it reloads -- same
    /// rule as `rorem_steps`.
    pub large_crop_ocr_px: Option<u32>,
    /// The ceiling on that routing, as a fraction of the page's own area, or
    /// `None` for no ceiling. Same reload rule as `large_crop_ocr_px`.
    pub large_crop_ocr_max_area: Option<f32>,
    /// Whether a region over that ceiling is read by no engine at all.
    pub skip_implausible_regions: bool,
    /// Whether a tall vertical free-text column is turned rather than widened.
    /// **ON by default**; see `--rotate-free-text-columns`.
    pub rotate_free_text_columns: bool,
    /// Whether a column the detector truncated at a slice edge is grown along its
    /// ink, and a sub-floor edge box reported as a hint. **ON by default**;
    /// see `--repair-clipped-columns`.
    pub repair_clipped_columns: bool,
    /// Whether a detected bubble holding no text region of its own is read as one.
    /// **ON by default**, adopted on rendered evidence; see
    /// `--read-textless-bubbles`.
    pub read_textless_bubbles: bool,
    /// `Some(floor)` lowers the `text` admission floor on JOINED pages only.
    /// **`Some(0.20)` by default**; see `--joined-page-text-floor`.
    pub joined_page_text_floor: Option<f32>,
    /// Whether NMS prefers the column-shaped box of a contained pair over the
    /// higher-scored one. **ON by default**, and SCOPED to declared zh/ko
    /// below; see `--axis-aware-nms`.
    pub axis_aware_nms: bool,
    /// Whether a box the axis tie-break evicts leaves its uncovered residue
    /// behind as its own region. **ON by default**; see
    /// `--nms-residue-regions`.
    pub nms_residue_regions: bool,
    /// Whether a read the pipeline already refuses to letter is kept out of the
    /// translation request. **ON by default**; see
    /// `--skip-unlettered-reads`.
    pub skip_unlettered_reads: bool,
    /// Whether an authored strike-through mark is re-drawn over the replacement
    /// lettering. **ON by default**; see
    /// `--strike-through-devices`.
    pub strike_through_devices: bool,
    /// Whether free-standing lettering takes fill, weight and outline from the
    /// drawn ink's sampled colour. **ON by default**; see
    /// `--sampled-ink-lettering`.
    pub sampled_ink_lettering: bool,
    /// Whether the free-text column turn is also offered on an unjoined page. ON
    /// is the shipped behaviour; see `--turn-unjoined-columns`.
    pub turn_unjoined_columns: bool,
    /// Whether a tall kana-free free-text column is read a second time turned 90
    /// CCW. **ON by default**, adopted on rendered pages; see
    /// `--reread-rotated-columns`.
    pub reread_rotated_columns: bool,
    /// How much more confident the upright read must be to beat the turned one.
    /// `None` disables the ranking; **the shipping value is `Some(0.17)`**; see
    /// `--orientation-confidence-margin`.
    pub orientation_confidence_margin: Option<f64>,
    /// Whether a free-standing region whose read came back empty is re-read
    /// through the sidecar's rotation sweep. **ON by default**; see
    /// `--upright-pass`.
    pub upright_pass: bool,
    /// Whether a SYNTHESISED bubble read is read a second time turned 180
    /// degrees, keeping the flipped read only when its confidence clearly wins.
    /// **ON by default**; see `--flip-reread-bubbles`.
    pub flip_reread_bubbles: bool,
    /// Whether a sparse or decline-carrying page may buy one spotting call and
    /// mint regions for display runs the detector never boxed. **ON by
    /// default**; see `--spot-rescue`.
    pub spot_rescue: bool,
    /// Whether a shipped spot rescue also joins the erase mask. **ON by
    /// default**; see `--spot-rescue-erase`.
    pub spot_rescue_erase: bool,
    /// Whether a wide scream-read mint becomes the mark-replacement device:
    /// ink-scoped erase, one styled gradient replacement on the ink's own
    /// axis. **ON by default**, adopted on rendered A/Bs; see
    /// `--replace-scream-marks`.
    pub replace_scream_marks: bool,
    /// Whether a seam composite may buy the spot call regardless of its region
    /// count. **ON by default**, adopted on a rendered chapter; see
    /// `--spot-rescue-joined`.
    pub spot_rescue_joined: bool,
    /// Pixels to grow an onomatopoeia crop by before reading it a SECOND time.
    /// **`None` disables it and is the shipping value**; reported, never acted on.
    /// See `--perturb-reread-grow-px`.
    pub perturb_reread_grow_px: Option<NonZeroU32>,
    /// Whether a watermark verdict is scoped to the site's own text rather than
    /// condemning the whole region. **ON by default**, adopted on rendered pages;
    /// see `--scope-watermark-refusals`. Written **twice**
    /// in `cli.rs`'s `resolve` -- this copy and the `defaults` one -- and a test now
    /// pins both, because nothing else made them agree.
    pub scope_watermark_refusals: bool,
    /// The pixel half of `--leave-misread-bubbles`: the script arm of
    /// `withdraw_from_mask` reaches dialogue-role regions too. Written twice in
    /// `resolve` exactly like `scope_watermark_refusals` above -- the `Shared`
    /// copy letters, this one erases -- and pinned by the same style of test.
    /// **ON by default.**
    pub leave_misread_bubbles: bool,
    /// The pixel half of `--korean-script-strict`. Same paired-write contract.
    /// **ON by default.**
    pub korean_script_strict: bool,
    /// Whether a lever-refused DIALOGUE read on declared ko buys ONE reserve
    /// re-read on a padded crop, admitted iff hangul-majority. **ON by
    /// default**; see `--reread-refused-dialogue`.
    pub reread_refused_dialogue: bool,
    /// Whether a region over that ceiling also contributes nothing to the erase
    /// mask, so the artwork under a mis-segmented box survives. Separate from
    /// `skip_implausible_regions` because "do not read it" and "do not erase it"
    /// are separate decisions and the A/B has to be able to name each end.
    pub skip_implausible_masks: bool,
    /// Take a region OCR could not read back out of the erase mask. The sibling
    /// of `skip_implausible_masks`, asked with the answer rather than a proxy.
    pub withdraw_illegible_masks: bool,
    /// Take a region no engine was ALLOWED to read back out of the erase mask.
    /// **ON by default**; breaks the interlock where
    /// `skip_implausible_regions` drops a box before `withdraw_illegible_masks` can
    /// ever see it. Measured: 18 of 18 cases fixed, 0 broken.
    pub withdraw_unread_masks: bool,
    /// Show a glyph cut by the inpaint tile grid to the model as one shape.
    pub seam_safe_erase: bool,
    /// `Some(dir)` writes the per-region detection masks and the assembled
    /// inpaint masks there as PNGs. Debug instrument; `None` ships.
    pub debug_mask_dir: Option<String>,
    /// Union the algorithmic ink mask into the erase mask.
    pub ink_mask: bool,
    /// Whether an eviction ends by returning torch's cached VRAM to the driver.
    ///
    /// Rides in `PipelineConfig` like `rorem_steps` does, so it is fixed for the
    /// process and a change would reload. That is the right shape here for a
    /// second reason as well: the flag exists to stop evictions, and a reload
    /// discards the residency profiles that decide when one happens, so flipping
    /// it mid-run would destroy the very state the measurement is about.
    pub release_cached_vram: Option<bool>,
    /// Whether a stage is asked for work before its weights are paged in, so a
    /// page with nothing to translate does not load the LLM. Same reload rule as
    /// `rorem_steps` -- it rides in `PipelineConfig.processor`.
    pub skip_empty_stages: bool,
}

/// Fixed at startup. See the module comment for why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pinned {
    pub provider: Provider,
    pub llm: String,
    /// The spelling the extension uses: "local" or "ollama".
    pub wire_provider: &'static str,
}

impl Pinned {
    /// Decides which LLM a request may actually use, or refuses it.
    ///
    /// An absent or empty `llm` means "whatever the server uses", which is what
    /// the popup sends when its Model box is blank. A *different* name is only
    /// honoured for `ollama`, where the weights are Ollama's problem rather than
    /// ours -- see the module comment.
    pub fn resolve(&self, provider: Option<&str>, llm: Option<&str>) -> Result<String, ApiError> {
        if provider.is_some_and(|value| value != self.wire_provider) {
            return Err(ApiError::new(
                StatusCode::CONFLICT,
                format!(
                    "provider/llm pinned to {}/{}; change the popup to match, or restart the server",
                    self.wire_provider,
                    clip(&self.llm, ECHO)
                ),
            ));
        }
        match llm.map(str::trim).filter(|value| !value.is_empty()) {
            None => Ok(self.llm.clone()),
            Some(value) if value == self.llm => Ok(self.llm.clone()),
            Some(value) if self.provider == Provider::OpenAiCompatible => Ok(value.to_owned()),
            Some(_) => Err(ApiError::new(
                StatusCode::CONFLICT,
                format!(
                    "llm pinned to {}; the local model cannot change without a restart",
                    clip(&self.llm, ECHO)
                ),
            )),
        }
    }
}

/// What one request is allowed to choose.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub ocr: OcrModel,
    pub inpainting: InpaintingModel,
    pub target_language: Language,
    /// What the page is written in, as `labels::SourceScript` already resolved it
    /// for this request -- the extension's per-host script latch, falling back to
    /// "manga-ocr implies Japanese".
    ///
    /// It is part of `Selection` rather than being read per page because it
    /// reaches the stages through `PipelineConfig`, and a config change reloads
    /// the pipeline. That is affordable only because the extension latches this
    /// **per host** and then leaves it alone; a value that moved every page would
    /// make every page a cold start.
    ///
    /// Feeds the OCR erase veto and nothing else. **Not the translator** --
    /// naming the source language in the prompt measured worse, over 511
    /// segments.
    ///
    /// Carried as the resolved `SourceScript` rather than as a `Language` so that
    /// `routes.rs` keeps exactly ONE resolution site. Its comment there is the
    /// reason: "a second resolution site is a second place for the two to
    /// disagree about what language the page is in."
    pub source_script: crate::labels::SourceScript,
    /// Already vetted by `Pinned::resolve`, so it is either the pinned name or
    /// an Ollama tag that costs this process no VRAM.
    pub llm: String,
    /// Whether the translator is told what each segment IS.
    ///
    /// On `Selection` for `source_script`'s reason and with the same caveat: it
    /// reaches the stages through `PipelineConfig`, so a change reloads the
    /// pipeline. Affordable because the extension carries it as a SETTING the
    /// reader flips occasionally, not a per-page control -- a value that moved
    /// every page would make every page a cold start.
    pub segment_context: bool,
}

/// Assembles the config exactly the way `run.rs` does, including the unset
/// quantization. `ModelSelection::default()` is deliberately not used: it names
/// a different model and does set a quantization.
#[must_use]
pub fn desired_config(
    selection: &Selection,
    defaults: &Defaults,
    pinned: &Pinned,
) -> PipelineConfig {
    PipelineConfig {
        detection: models::detection(
            defaults.translate_sfx,
            defaults.refine_text_mask,
            defaults.onomatopoeia_threshold,
            defaults.mask_scale,
            defaults.ink_mask,
        ),
        ocr: selection.ocr.clone(),
        translation: TranslationConfig {
            model: ModelSelection {
                provider: pinned.provider,
                model: Some(selection.llm.clone()),
                quantization: None,
            },
            generation: GenerationConfig {
                temperature: defaults.temperature,
                swa_full: defaults.swa_full,
                ..GenerationConfig::default()
            },
            target_language: selection.target_language,
            source_language: selection.source_script.language(),
            instructions: defaults.instructions.clone(),
            containment_clause: defaults.containment_clause,
            segment_context: selection.segment_context,
        },
        inpainting: selection.inpainting.clone(),
        processor: koharu_pipeline::ProcessorConfig {
            large_crop_ocr_px: defaults.large_crop_ocr_px,
            large_crop_ocr_max_area: defaults.large_crop_ocr_max_area,
            skip_implausible_regions: defaults.skip_implausible_regions,
            rotate_free_text_columns: defaults.rotate_free_text_columns,
            repair_clipped_columns: defaults.repair_clipped_columns,
            read_textless_bubbles: defaults.read_textless_bubbles,
            joined_page_text_floor: defaults.joined_page_text_floor,
            /* SCOPED BY DECLARED SCRIPT, chosen on a two-corpus census: a
             * clear win on a zh webtoon chapter (a rescued dialogue line,
             * zero losses) and a clear loss on ja manga (10 of 91 pages
             * degraded -- stray un-erased glyphs, a name misread). The
             * column-shape prior is right for CJK display columns and wrong
             * for Japanese dialogue boxes -- the same ja/non-ja line the OCR
             * engines are split on, resolved in the same layer. `Unknown`
             * counts as Japanese here ON PURPOSE: an undeclared request
             * through a non-ja engine can still be Japanese
             * (labels.rs::SourceScript::resolve carries the measurement), and
             * firing on it would re-run the refuted arm. Korean is included
             * but UNCENSUSED -- it has no measured arm of its own. */
            axis_aware_nms: defaults.axis_aware_nms
                && matches!(
                    selection.source_script,
                    crate::labels::SourceScript::Chinese | crate::labels::SourceScript::Korean
                ),
            /* NOT separately script-scoped, and it does not need to be:
             * it acts only where the tie-break above already evicted a
             * box, and that is scoped on the line above. Scoping it twice
             * would read as a second, independent gate and invite someone
             * to relax one of them. */
            nms_residue_regions: defaults.nms_residue_regions,
            /* NOT script-scoped either, and for a stronger reason than its
             * neighbour: every arm of `unlettered_read` is language-independent
             * by construction -- the OCR stage's own veto, an illegible read,
             * and a watermark literal. The script-dependent refusals stay
             * downstream where the declared language is known. */
            skip_unlettered_reads: defaults.skip_unlettered_reads,
            /* NOT script-scoped. The device is a drawn mark, not a language
             * feature: the detector measures ink geometry and never asks what
             * the text says, so scoping it to zh/ko would be superstition
             * rather than a measured ja/non-ja difference. */
            strike_through_devices: defaults.strike_through_devices,
            /* SCOPED BY DECLARED SCRIPT — by design, Japanese text is never
             * turned: it is lettered across, in a fitting area that hides
             * little of the artwork. The unjoined turn's own doc
             * (detection.rs, `turn_unjoined_columns`) predicted this class in
             * writing — 38.3% of manga-ja free-text is ordinary upright
             * vertical Japanese, and turning those would letter English sideways
             * down a column that should read across — and the render the
             * turn was adopted on priced that on TWO sound effects (~0.1% of
             * two pages) because it had no dense-ja text page to see. A dense
             * Japanese page rendered 5 of 5 fields SIDEWAYS; the off arm
             * letters all five horizontal in the widened cell, both looked
             * at. Declared zh/ko keep the turn, adopted on renders;
             * `Unknown` counts as Japanese for the same measured reason
             * `axis_aware_nms`'s scope above records, and the popup's
             * per-site language card is how a reader declares when the latch
             * has not. The JOINED arm (`rotate_free_text_columns` on a
             * `joined_page`) is deliberately untouched — its exemplars are
             * declared-zh composites; a declared-ja JOINED page still turning
             * is a known residual, not a case this scope reaches. */
            turn_unjoined_columns: defaults.turn_unjoined_columns
                && matches!(
                    selection.source_script,
                    crate::labels::SourceScript::Chinese | crate::labels::SourceScript::Korean
                ),
            /* UNSCOPED by script on purpose: drawn screams and effects exist in
             * every corpus, the override keys on the free-standing role rather
             * than a language, and a region whose ink the erosion cannot read
             * abstains into today's behaviour on its own. */
            sampled_ink_lettering: defaults.sampled_ink_lettering,
            reread_rotated_columns: defaults.reread_rotated_columns,
            orientation_confidence_margin: defaults.orientation_confidence_margin,
            upright_pass: defaults.upright_pass,
            flip_reread_bubbles: defaults.flip_reread_bubbles,
            spot_rescue: defaults.spot_rescue,
            spot_rescue_erase: defaults.spot_rescue_erase,
            /* UNSCOPED by script for the same reason as the sampler above:
             * drawn scream marks exist in every corpus, the predicate keys on
             * the mint's own geometry and read, and an abstention ships
             * today's behaviour. */
            replace_scream_marks: defaults.replace_scream_marks,
            spot_rescue_joined: defaults.spot_rescue_joined,
            perturb_reread_grow_px: defaults.perturb_reread_grow_px,
            scope_watermark_refusals: defaults.scope_watermark_refusals,
            leave_misread_bubbles: defaults.leave_misread_bubbles,
            korean_script_strict: defaults.korean_script_strict,
            reread_refused_dialogue: defaults.reread_refused_dialogue,
            skip_implausible_masks: defaults.skip_implausible_masks,
            withdraw_illegible_masks: defaults.withdraw_illegible_masks,
            withdraw_unread_masks: defaults.withdraw_unread_masks,
            seam_safe_erase: defaults.seam_safe_erase,
            debug_mask_dir: defaults.debug_mask_dir.clone(),
            release_cached_vram: defaults.release_cached_vram,
            skip_empty_stages: defaults.skip_empty_stages,
            ..Default::default()
        },
    }
}

/// An unconditional reload is a real cost, not a harmless no-op: it discards the
/// warm detection, OCR and inpainting weights on every single request.
#[must_use]
pub fn needs_reload(applied: &PipelineConfig, desired: &PipelineConfig) -> bool {
    applied != desired
}

/// Installs `desired` if it differs from what is running. Must be called while
/// holding the gate.
pub async fn reconcile(
    pipeline: &Pipeline,
    gpu: &mut GpuState,
    desired: PipelineConfig,
) -> Result<(), ApiError> {
    if !needs_reload(&gpu.applied, &desired) {
        return Ok(());
    }
    tracing::info!("reloading the pipeline for a changed model selection");
    pipeline.reload(&desired).map_err(|error| {
        ApiError::internal(crate::error::clip(
            &format!("could not apply the requested models: {error}"),
            crate::error::CLIP_BUDGET,
        ))
    })?;
    gpu.applied = desired;
    tokio::time::sleep(SETTLE_AFTER_RELOAD).await;
    Ok(())
}

/// Whether anything is resident, for `/status`, `/warmup` and the pre-flight.
///
/// Translation counts only under the local engine. koharu's translator reports
/// itself loaded unconditionally for every remote provider -- correctly, since
/// there is nothing of ours to load -- so folding it in regardless would make a
/// stone-cold `--provider ollama` server look warm, `/warmup` a permanent no-op
/// and the popup's countdown meaningless.
///
/// A stage whose model cell is held by a run in flight reports loaded, which is
/// what every caller here wants: busy is a kind of loaded.
#[must_use]
pub fn models_loaded(state: &Shared) -> bool {
    const DEVICE_STAGES: [Stage; 3] = [Stage::Detection, Stage::Ocr, Stage::Inpainting];
    DEVICE_STAGES
        .iter()
        .any(|stage| state.pipeline.model_loaded(*stage))
        || (state.pinned.provider == Provider::Local
            && state.pipeline.model_loaded(Stage::Translation))
}

/// Turns a failed stage into the 500 the extension can read.
///
/// The stage name leads because it is the one word that says where the run died,
/// and after it there is room for exactly one line. That line is the innermost
/// cause rather than koharu's head -- see `error::root_cause` -- and the whole
/// chain goes to the log, which is the only place with room for it.
#[must_use]
pub fn stage_failure(error: &PipelineError) -> ApiError {
    tracing::error!(
        stage = ?error.stage,
        kind = ?error.kind,
        chain = %crate::error::chain(error),
        "a pipeline stage failed"
    );
    let stage = error
        .stage
        .map_or_else(|| "the pipeline".to_owned(), |stage| stage.to_string());
    ApiError::internal(clip(
        &format!("{stage} failed: {}", root_cause(error)),
        CLIP_BUDGET,
    ))
}

/// Drops every model's weights, reporting whether anything was actually freed.
/// Must be called while holding the gate.
///
/// The gate is what makes it correct rather than merely likely: koharu's unload
/// gives up on a model cell that is locked instead of waiting for it, and reports
/// the same `false` it reports for a cell that was already empty.
///
/// `unload_models` already reaches the local LLM through the translation stage,
/// so the second call is expected to report `false`. It stays because the
/// translator is the only direct handle on those weights if that ever changes,
/// and it costs one `try_lock` on an empty cell.
///
/// Caught, because this is the one place the server runs a model's destructor,
/// and both of its callers are places a panic must not reach. `/unload` is an
/// axum handler, where an escaping panic costs the caller its connection instead
/// of a response; `idle::watch` is spawned and never joined, so a panic there
/// would stop the idle budget being enforced for the rest of the process without
/// anyone being told. A panic reports `false`, which is what every caller already
/// treats as "nothing was freed", and it is deliberately not retried.
#[must_use]
pub fn free_models(pipeline: &Pipeline) -> bool {
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let stages = pipeline.unload_models();
        let llm = pipeline.translator().unload();
        stages || llm
    }))
    .unwrap_or_else(|payload| {
        tracing::error!(panic = %panic_text(&*payload), "freeing the models panicked");
        false
    })
}

/// Loads every model off the request path.
///
/// Holds the gate for the whole load, so a page arriving meanwhile queues behind
/// it rather than racing it into the same VRAM.
pub async fn warm_models(state: AppState) {
    let _warming = WarmupInFlight(state.clone());
    let outcome = {
        let _gpu = state.gate.lock().await;
        // Caught rather than left to escape: nothing joins this task, so a panic
        // in a model load would be reported to no one, and the gate has to be
        // released by an ordinary return from this block.
        AssertUnwindSafe(state.pipeline.load_models())
            .catch_unwind()
            .await
            .unwrap_or_else(|payload| {
                Err(anyhow::anyhow!(
                    "the model load panicked: {}",
                    panic_text(&*payload)
                ))
            })
    };
    // Started on both paths: a load that failed part way through still leaves
    // the stages before it resident, and those still belong to the idle budget.
    state.idle.touch();
    match outcome {
        Ok(()) => tracing::info!("warmup finished; the models are resident"),
        Err(error) => tracing::error!(?error, "warmup failed"),
    }
}

/// Releases the warmup claim however the load ends. A plain store at the end
/// would leave `/warmup` answering "already loading" for the rest of the process
/// if the load panicked instead of returning an error.
struct WarmupInFlight(AppState);

impl Drop for WarmupInFlight {
    fn drop(&mut self) {
        self.0.warming.store(false, Ordering::Release);
    }
}

/// Belt and braces for the reload: the stage runner reports the model that
/// actually ran, so a config that failed to land is visible after the fact.
/// With a synchronous reload this can never fire; if it ever does, the reload
/// path regressed. Only detection, OCR and inpainting are reported -- the
/// target language and the LLM name are not, which is exactly why they are not
/// applied through the watcher.
pub fn verify_applied(observed: &[(Stage, String)], desired: &PipelineConfig) {
    for (stage, model) in observed {
        let expected = match stage {
            Stage::Ocr => models::ocr_model_name(&desired.ocr),
            Stage::Inpainting => models::inpainting_model_name(&desired.inpainting),
            _ => continue,
        };
        if model != expected {
            tracing::error!(%stage, ran = %model, wanted = %expected, "stale stage runner");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::BUDGET;

    fn pinned_ollama() -> Pinned {
        Pinned {
            provider: Provider::OpenAiCompatible,
            llm: "qwen3:8b".to_owned(),
            wire_provider: models::PROVIDER_OLLAMA,
        }
    }

    fn pinned_local() -> Pinned {
        Pinned {
            provider: Provider::Local,
            llm: models::DEFAULT_LOCAL_MODEL.to_owned(),
            wire_provider: models::PROVIDER_LOCAL,
        }
    }

    fn defaults() -> Defaults {
        Defaults {
            ocr: OcrModel::PaddleOcrVl1_6,
            ocr_substitute: None,
            inpainting: InpaintingModel::LaMa {},
            target_language: Language::English,
            instructions: None,
            containment_clause: false,
            segment_context: false,
            temperature: None,
            swa_full: None,
            translate_sfx: true,
            rotate_free_text_columns: false,
            repair_clipped_columns: false,
            read_textless_bubbles: false,
            joined_page_text_floor: None,
            axis_aware_nms: false,
            nms_residue_regions: false,
            skip_unlettered_reads: false,
            strike_through_devices: false,
            sampled_ink_lettering: false,
            turn_unjoined_columns: false,
            reread_rotated_columns: false,
            orientation_confidence_margin: None,
            upright_pass: false,
            flip_reread_bubbles: false,
            spot_rescue: false,
            spot_rescue_erase: false,
            replace_scream_marks: false,
            spot_rescue_joined: false,
            perturb_reread_grow_px: None,
            scope_watermark_refusals: false,
            leave_misread_bubbles: false,
            korean_script_strict: false,
            reread_refused_dialogue: false,
            refine_text_mask: false,
            rorem_steps: None,
            flux_strength: None,
            onomatopoeia_threshold: None,
            mask_scale: None,
            large_crop_ocr_px: None,
            large_crop_ocr_max_area: None,
            skip_implausible_regions: false,
            skip_implausible_masks: false,
            // False here on purpose, like the two above it: this fixture exists
            // to describe a pipeline with every optional gate OFF, so a test
            // asserting a default has something to differ from.
            withdraw_illegible_masks: false,
            withdraw_unread_masks: false,
            seam_safe_erase: false,
            debug_mask_dir: None,
            ink_mask: false,
            release_cached_vram: None,
            // The `ProcessorConfig` default, so `the_config_matches_the_headless_cli`
            // keeps comparing against `Default::default()`. The *server's* default
            // is `true`; that is asserted in `cli.rs`, where the flag lives.
            skip_empty_stages: false,
        }
    }

    fn selection() -> Selection {
        Selection {
            ocr: OcrModel::PaddleOcrVl1_6,
            inpainting: InpaintingModel::LaMa {},
            target_language: Language::English,
            source_script: crate::labels::SourceScript::Unknown,
            llm: models::DEFAULT_LOCAL_MODEL.to_owned(),
            segment_context: false,
        }
    }

    #[test]
    fn a_matching_request_passes() {
        assert_eq!(
            pinned_local().resolve(Some("local"), None).unwrap(),
            models::DEFAULT_LOCAL_MODEL
        );
    }

    #[test]
    fn a_blank_model_box_means_the_servers_own_model() {
        // The popup omits `llm` entirely when its field is empty, and sends an
        // empty string if it ever stops doing so.
        for llm in [None, Some(""), Some("  ")] {
            assert_eq!(
                pinned_ollama().resolve(Some("ollama"), llm).unwrap(),
                "qwen3:8b",
                "{llm:?}"
            );
        }
    }

    #[test]
    fn a_different_provider_is_a_conflict() {
        let error = pinned_ollama().resolve(Some("local"), None).unwrap_err();
        assert_eq!(error.status, StatusCode::CONFLICT);
        assert!(error.message.len() <= BUDGET);
        // Names the provider to pick, which is the whole point: the extension
        // defaults to "ollama" while the server defaults to "local".
        assert!(error.message.contains("ollama"));
        assert!(error.message.contains("qwen3:8b"));
    }

    #[test]
    fn a_different_model_is_a_conflict_only_for_the_local_engine() {
        let error = pinned_local()
            .resolve(Some("local"), Some("gemma4-12b-it"))
            .unwrap_err();
        assert_eq!(error.status, StatusCode::CONFLICT);
        assert!(error.message.len() <= BUDGET);
    }

    #[test]
    fn ollama_may_name_any_tag_it_likes() {
        // Ollama holds those weights in its own process, so this costs us no
        // VRAM and needs no restart -- only a stage-runner reload.
        assert_eq!(
            pinned_ollama()
                .resolve(Some("ollama"), Some("qwen3:14b"))
                .unwrap(),
            "qwen3:14b"
        );
    }

    #[test]
    fn the_conflict_message_survives_a_very_long_pinned_model() {
        let long = "m".repeat(4096);
        for error in [
            Pinned {
                llm: long.clone(),
                ..pinned_ollama()
            }
            .resolve(Some("local"), None)
            .unwrap_err(),
            Pinned {
                llm: long,
                ..pinned_local()
            }
            .resolve(Some("local"), Some("other"))
            .unwrap_err(),
        ] {
            let message = error.message;
            assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
        }
    }

    #[test]
    fn the_config_matches_the_headless_cli() {
        let config = desired_config(&selection(), &defaults(), &pinned_local());
        assert!(matches!(
            config.detection,
            koharu_pipeline::DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert_eq!(config.translation.model.quantization, None);
        assert_eq!(
            config.translation.model.model.as_deref(),
            Some(models::DEFAULT_LOCAL_MODEL)
        );
        assert_eq!(config.translation.model.provider, Provider::Local);
        assert_eq!(config.processor, Default::default());
    }

    #[test]
    fn only_a_real_change_forces_a_reload() {
        let defaults = defaults();
        let pinned = pinned_local();
        let applied = desired_config(&selection(), &defaults, &pinned);
        assert!(!needs_reload(&applied, &applied.clone()));

        let switched = desired_config(
            &Selection {
                target_language: Language::Korean,
                ..selection()
            },
            &defaults,
            &pinned,
        );
        assert!(needs_reload(&applied, &switched));
    }

    /// The wiring proof for the source-language plumbing: a declared script must
    /// actually arrive in `PipelineConfig`, and must be seen as a real change.
    ///
    /// **This asserts the field's VALUE as well as the reload**, because the two
    /// fail independently. A `needs_reload` that returns true proves only that
    /// something differs; without the value assertion the whole chain could carry
    /// the wrong language and the test would still be green.
    ///
    /// `Unknown -> None` is asserted too. That direction is the one that matters
    /// for safety: inferring a language from silence would fire the kana rule on
    /// genuine Japanese dialogue and drop it.
    #[test]
    fn a_declared_source_script_reaches_the_pipeline_config_and_forces_a_reload() {
        let defaults = defaults();
        let pinned = pinned_local();

        let unknown = desired_config(&selection(), &defaults, &pinned);
        assert_eq!(
            unknown.translation.source_language, None,
            "an undeclared script must reach the pipeline as None, or nothing-fires-on-silence breaks"
        );

        let chinese = desired_config(
            &Selection {
                source_script: crate::labels::SourceScript::Chinese,
                ..selection()
            },
            &defaults,
            &pinned,
        );
        assert_eq!(
            chinese.translation.source_language,
            "zh".parse().ok(),
            "the declared script must arrive as a Language, not merely differ"
        );
        assert!(needs_reload(&unknown, &chinese));

        // Japanese is the case the erase veto must never fire on, so prove it
        // still travels rather than being silently mapped to None.
        let japanese = desired_config(
            &Selection {
                source_script: crate::labels::SourceScript::Japanese,
                ..selection()
            },
            &defaults,
            &pinned,
        );
        assert_eq!(
            japanese.translation.source_language,
            Some(Language::Japanese)
        );
        assert!(needs_reload(&chinese, &japanese));
    }

    /// Oversized-crop routing rides on `PipelineConfig.processor`, so toggling it
    /// must reload. If this stops holding, a reader turning it on would keep the
    /// old stage runner and see no change -- the silent-no-op failure that
    /// `verify_applied` exists to catch for models.
    #[test]
    fn changing_the_large_crop_threshold_forces_a_reload() {
        let pinned = pinned_local();
        let off = desired_config(&selection(), &defaults(), &pinned);
        let on = desired_config(
            &selection(),
            &Defaults {
                large_crop_ocr_px: Some(448),
                ..defaults()
            },
            &pinned,
        );
        assert!(needs_reload(&off, &on));
        assert_eq!(on.processor.large_crop_ocr_px, Some(448));
        assert_eq!(off.processor.large_crop_ocr_px, None);
    }

    /// The ceiling rides in the same `PipelineConfig.processor` as the threshold
    /// it bounds, so both arms of the A/B have to be distinguishable there. A
    /// ceiling that reached `StageRunner::new` on only one arm would look like a
    /// null result rather than a flag that never arrived.
    #[test]
    fn both_arms_of_the_region_area_ceiling_reach_the_pipeline_config() {
        let pinned = pinned_local();
        let off = desired_config(&selection(), &defaults(), &pinned);
        let on = desired_config(
            &selection(),
            &Defaults {
                large_crop_ocr_max_area: Some(0.5),
                skip_implausible_regions: true,
                ..defaults()
            },
            &pinned,
        );
        assert!(needs_reload(&off, &on));
        assert_eq!(on.processor.large_crop_ocr_max_area, Some(0.5));
        assert!(on.processor.skip_implausible_regions);
        assert_eq!(off.processor.large_crop_ocr_max_area, None);
        assert!(!off.processor.skip_implausible_regions);
    }

    /// The joined-floor and tie-break pair rides in `PipelineConfig.processor`,
    /// so both halves have to be distinguishable there. `desired_config`'s
    /// processor literal ends in
    /// `..Default::default()`, which is exactly how a resolved flag can silently
    /// fail to arrive: a missing mapping line compiles, defaults, and reads as
    /// "the fix did nothing". A CHINESE selection, because the tie-break is
    /// script-scoped and this test is about the wiring, not the scope.
    #[test]
    fn both_arms_of_the_sampled_ink_lettering_reach_the_pipeline_config() {
        let pinned = pinned_local();
        let off = desired_config(&selection(), &defaults(), &pinned);
        let on = desired_config(
            &selection(),
            &Defaults {
                sampled_ink_lettering: true,
                ..defaults()
            },
            &pinned,
        );
        assert!(needs_reload(&off, &on));
        assert!(on.processor.sampled_ink_lettering);
        assert!(!off.processor.sampled_ink_lettering);
    }

    /// THE POPUP TOGGLE REACHES THE PROMPT, and the SELECTION is what carries it.
    ///
    /// This read `defaults.segment_context` until the reader got a switch.
    /// `desired_config` now takes the value off the per-request `Selection`, and
    /// `selection_from` in `routes.rs` is what falls back to the process default
    /// when the caller sends no field. A test still pointed at `Defaults` would
    /// pass on a build where the toggle reached nothing at all -- which is what
    /// this one did, red, the moment the lever moved.
    ///
    /// It also pins the RELOAD, which is the toggle's whole cost: flipping it
    /// makes `applied != desired`, so the pipeline rebuilds exactly as it does
    /// for an OCR or inpainting change. Affordable for a setting, and not for a
    /// per-page control -- which is why the popup presents it as the former.
    #[test]
    fn the_segment_context_toggle_reaches_the_prompt_through_the_selection() {
        let pinned = pinned_local();
        let off = desired_config(&selection(), &defaults(), &pinned);
        let on = desired_config(
            &Selection {
                segment_context: true,
                ..selection()
            },
            &defaults(),
            &pinned,
        );
        assert!(
            !off.translation.segment_context,
            "a request that asks for nothing must not describe its segments"
        );
        assert!(
            on.translation.segment_context,
            "the reader's toggle must actually reach the translator"
        );
        assert!(
            needs_reload(&off, &on),
            "and flipping it must reload: otherwise the prompt changes while the \
             pipeline that serves it does not"
        );
    }

    /// The strike device's own wire, for the residue test's reason below. This
    /// also pins that it is NOT script-scoped: a drawn mark is ink geometry, not
    /// a language feature, so a zh/ko gate here would be superstition rather than
    /// a measured ja/non-ja difference. The fixture is deliberately JAPANESE for
    /// that reason.
    #[test]
    fn the_strike_device_reaches_the_pipeline_config_and_is_not_script_scoped() {
        let pinned = pinned_local();
        let japanese = Selection {
            source_script: crate::labels::SourceScript::Japanese,
            ..selection()
        };
        let off = desired_config(&japanese, &defaults(), &pinned);
        let on = desired_config(
            &japanese,
            &Defaults {
                strike_through_devices: true,
                ..defaults()
            },
            &pinned,
        );
        assert!(
            !off.processor.strike_through_devices,
            "an off arm must reach the pipeline as off -- note this is the              test-local `defaults()`, NOT the shipping default, which is              ON; the shipping value is asserted in cli.rs"
        );
        assert!(
            on.processor.strike_through_devices,
            "asking for it must deliver it, on Japanese as much as on Chinese"
        );
    }

    /// THE FLAG REACHES THE PIPELINE, which is a separate claim from "the
    /// residue logic works" and the one that is cheap to get wrong.
    ///
    /// `settle_detections` has its own on/off pair asserting the behaviour;
    /// what neither of those can see is a flag that never arrives. A fix has
    /// shipped on this project completely unwired with every test green, so the
    /// wire gets its own assertion rather than being inferred from the unit.
    ///
    /// It does NOT pin the shipping default, and used to claim it did: the arms
    /// here are built from the test-local `defaults()`, which is all-false, so
    /// this test reads the same whatever `Cli::resolve` decides. The shipping
    /// value is ON and is asserted in `cli.rs`; `false`
    /// remains the byte-exact control arm the tie-break needs.
    #[test]
    fn the_residue_admission_reaches_the_pipeline_config_and_keeps_a_control_arm() {
        let pinned = pinned_local();
        let chinese = Selection {
            source_script: crate::labels::SourceScript::Chinese,
            ..selection()
        };
        let off = desired_config(&chinese, &defaults(), &pinned);
        let on = desired_config(
            &chinese,
            &Defaults {
                axis_aware_nms: true,
                nms_residue_regions: true,
                ..defaults()
            },
            &pinned,
        );
        assert!(
            !off.processor.nms_residue_regions,
            "an off arm must reach the pipeline as off -- note this is the              test-local `defaults()`, NOT the shipping default, which is              ON; the shipping value is asserted in cli.rs"
        );
        assert!(
            on.processor.nms_residue_regions,
            "asking for it must actually deliver it to the pipeline"
        );
        assert!(needs_reload(&off, &on));
    }

    /// The residue flag's Japanese safety is **transitive, not enforced here**,
    /// and this test pins the exact shape that argument rests on.
    ///
    /// It matters because the flag ships ON: unlike
    /// its tie-break, it is not script-scoped, so on a declared-Japanese request
    /// the config field really is `true`. It is inert anyway, because a residue
    /// exists only where the tie-break EVICTED a box and the tie-break never
    /// fires here. If a later change ever admits a residue without an eviction,
    /// this is the test that should have stopped it -- the ja arm of
    /// `axis_aware_nms` is refuted (10 of 91 pages of a Japanese test volume
    /// degraded), so a residue reaching Japanese re-runs an arm the census
    /// already killed.
    #[test]
    fn the_residue_flag_is_inert_on_japanese_only_because_the_tie_break_is_scoped() {
        let pinned = pinned_local();
        let shipping = Defaults {
            axis_aware_nms: true,
            nms_residue_regions: true,
            ..defaults()
        };
        let japanese = Selection {
            source_script: crate::labels::SourceScript::Japanese,
            ..selection()
        };
        let cfg = desired_config(&japanese, &shipping, &pinned);
        assert!(
            !cfg.processor.axis_aware_nms,
            "the ja arm is refuted -- the tie-break must not fire on Japanese"
        );
        assert!(
            cfg.processor.nms_residue_regions,
            "the residue flag is NOT itself scoped; it rides the line above, and              this assertion is what says that dependency out loud"
        );
    }

    #[test]
    fn both_halves_of_the_floor_and_tie_break_pair_reach_the_pipeline_config() {
        let pinned = pinned_local();
        let chinese = Selection {
            source_script: crate::labels::SourceScript::Chinese,
            ..selection()
        };
        let off = desired_config(&chinese, &defaults(), &pinned);
        let on = desired_config(
            &chinese,
            &Defaults {
                joined_page_text_floor: Some(0.20),
                axis_aware_nms: true,
                ..defaults()
            },
            &pinned,
        );
        assert!(needs_reload(&off, &on));
        assert_eq!(on.processor.joined_page_text_floor, Some(0.20));
        assert!(on.processor.axis_aware_nms);
        assert_eq!(off.processor.joined_page_text_floor, None);
        assert!(!off.processor.axis_aware_nms);
        // The control arm (floor alone) and the paired arm are different
        // configs, or the census could not run one against the other.
        let control = desired_config(
            &chinese,
            &Defaults {
                joined_page_text_floor: Some(0.20),
                ..defaults()
            },
            &pinned,
        );
        assert!(needs_reload(&control, &on));
    }

    /// The scope chosen on the census: the tie-break fires only on a
    /// POSITIVELY declared Chinese or Korean script. Japanese is the refuted
    /// arm; `Unknown` counts as Japanese because an undeclared request through
    /// a non-ja engine can still be Japanese, and firing on it would re-run
    /// the refutation. An explicit `false` wins everywhere.
    #[test]
    fn the_tie_break_is_scoped_to_declared_non_japanese_scripts() {
        let pinned = pinned_local();
        let on = Defaults {
            axis_aware_nms: true,
            ..defaults()
        };
        for (script, expected) in [
            (crate::labels::SourceScript::Chinese, true),
            (crate::labels::SourceScript::Korean, true),
            (crate::labels::SourceScript::Japanese, false),
            (crate::labels::SourceScript::Unknown, false),
        ] {
            let config = desired_config(
                &Selection {
                    source_script: script,
                    ..selection()
                },
                &on,
                &pinned,
            );
            assert_eq!(
                config.processor.axis_aware_nms, expected,
                "script {script:?} must resolve the tie-break to {expected}"
            );
            // The flag's own off arm is not scoped: false everywhere.
            let off = desired_config(
                &Selection {
                    source_script: script,
                    ..selection()
                },
                &defaults(),
                &pinned,
            );
            assert!(!off.processor.axis_aware_nms);
        }
    }

    /// The never-rotate-Japanese rule operationalized: the unjoined
    /// column turn fires only on a POSITIVELY declared Chinese or Korean
    /// script — the display-column population it was measured on. Japanese is
    /// the population the rule protects (upright vertical dialogue columns,
    /// 38.3% of manga-ja free-text per detection.rs's own doc), and `Unknown`
    /// counts as Japanese for the same reason the tie-break's scope gives: an
    /// undeclared request can still be Japanese, and a sideways page is the
    /// worse error. An explicit `false` wins everywhere.
    #[test]
    fn the_unjoined_turn_is_scoped_to_declared_non_japanese_scripts() {
        let pinned = pinned_local();
        let on = Defaults {
            turn_unjoined_columns: true,
            ..defaults()
        };
        let off = Defaults {
            turn_unjoined_columns: false,
            ..defaults()
        };
        for (script, expected) in [
            (crate::labels::SourceScript::Chinese, true),
            (crate::labels::SourceScript::Korean, true),
            (crate::labels::SourceScript::Japanese, false),
            (crate::labels::SourceScript::Unknown, false),
        ] {
            let config = desired_config(
                &Selection {
                    source_script: script,
                    ..selection()
                },
                &on,
                &pinned,
            );
            assert_eq!(
                config.processor.turn_unjoined_columns, expected,
                "script {script:?} must resolve the unjoined turn to {expected}"
            );
            let config = desired_config(
                &Selection {
                    source_script: script,
                    ..selection()
                },
                &off,
                &pinned,
            );
            assert!(
                !config.processor.turn_unjoined_columns,
                "an explicit false wins under {script:?}"
            );
        }
    }

    /// The action is a separate decision from the bound, so flipping it alone
    /// must still reload -- otherwise a reader turning it on would keep the old
    /// stage runner and see the conservative arm while believing otherwise.
    #[test]
    fn the_skip_arm_reloads_on_its_own() {
        let pinned = pinned_local();
        let bounded = Defaults {
            large_crop_ocr_max_area: Some(0.5),
            ..defaults()
        };
        let refuse_only = desired_config(&selection(), &bounded, &pinned);
        let skip = desired_config(
            &selection(),
            &Defaults {
                skip_implausible_regions: true,
                ..bounded.clone()
            },
            &pinned,
        );
        assert!(needs_reload(&refuse_only, &skip));
    }

    /// The mask arm is a third decision, not a rename of the second. "Do not
    /// read it" and "do not erase it" have to be independently settable in
    /// `PipelineConfig`, because the whole point of the pair is that only the
    /// second one can save the artwork under a phantom box -- and an
    /// A/B that could not name them apart would credit one for the other.
    #[test]
    fn the_mask_arm_is_independent_of_the_skip_arm() {
        let pinned = pinned_local();
        let bounded = Defaults {
            large_crop_ocr_max_area: Some(0.5),
            skip_implausible_regions: true,
            ..defaults()
        };
        let read_only = desired_config(&selection(), &bounded, &pinned);
        let masked = desired_config(
            &selection(),
            &Defaults {
                skip_implausible_masks: true,
                ..bounded.clone()
            },
            &pinned,
        );
        assert!(needs_reload(&read_only, &masked));
        assert!(!read_only.processor.skip_implausible_masks);
        assert!(masked.processor.skip_implausible_masks);
        // Still true on both arms: turning the mask gate on must not quietly
        // change what OCR does with the same box.
        assert!(read_only.processor.skip_implausible_regions);
        assert!(masked.processor.skip_implausible_regions);
    }

    /// The A/B arms have to be distinguishable in `PipelineConfig` itself, or the
    /// flag reaches `StageRunner::new` on one arm and not the other only by
    /// accident of startup ordering.
    #[test]
    fn both_arms_of_the_cached_vram_release_reach_the_pipeline_config() {
        let pinned = pinned_local();
        let unset = desired_config(&selection(), &defaults(), &pinned);
        let on = desired_config(
            &selection(),
            &Defaults {
                release_cached_vram: Some(true),
                ..defaults()
            },
            &pinned,
        );
        let off = desired_config(
            &selection(),
            &Defaults {
                release_cached_vram: Some(false),
                ..defaults()
            },
            &pinned,
        );
        assert_eq!(unset.processor.release_cached_vram, None);
        assert_eq!(on.processor.release_cached_vram, Some(true));
        assert_eq!(off.processor.release_cached_vram, Some(false));
        assert!(needs_reload(&on, &off));
    }

    /// Same rule as every other `processor` flag: both arms must be visible in
    /// `PipelineConfig`, or the gate reaches `StageRunner::new` on one arm only
    /// by accident of startup ordering and the A/B measures nothing.
    #[test]
    fn both_arms_of_the_empty_stage_gate_reach_the_pipeline_config() {
        let pinned = pinned_local();
        let off = desired_config(&selection(), &defaults(), &pinned);
        let on = desired_config(
            &selection(),
            &Defaults {
                skip_empty_stages: true,
                ..defaults()
            },
            &pinned,
        );
        assert!(!off.processor.skip_empty_stages);
        assert!(on.processor.skip_empty_stages);
        assert!(needs_reload(&off, &on));
    }
}
