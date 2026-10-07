//! The endpoints the extension talks to.
//!
//! `/health` is the only one outside the token layer, because the popup's ping
//! sends no headers at all and only reads whether the response was ok.

use std::{
    any::Any,
    panic::AssertUnwindSafe,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, State, multipart::MultipartRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware,
    response::{IntoResponse, Response},
    routing::{MethodRouter, get, post},
};
use base64::Engine as _;
use futures::FutureExt as _;
use koharu_pipeline::{Operation, PipelineConfig, Progress, ProgressSink, Request, Scope, Stage};
use koharu_scene::{EntityId, Session, Snapshot};
use serde::Serialize;
use tower_http::trace::TraceLayer;

use crate::{
    engine::{
        AppState, Defaults, GpuState, Pinned, Selection, desired_config, free_models,
        models_loaded, needs_reload, reconcile, stage_failure, verify_applied, warm_models,
    },
    error::{
        ApiError, CLIP_BUDGET, MISSING_IMAGE, RENDER_TASK_FAILED, UNLOAD_BUSY, clip, panic_text,
    },
    form::{TranslateForm, read_form, rejection_error, wants_json},
    guard::{GuardState, guard_origin_host, require_token},
    idle::TouchOnDrop,
    models,
    regions::{RegionOut, regions},
    scene::{PreparedPage, SessionCommitter, prepare_page},
    vram::{self, Vram},
};

#[derive(Serialize)]
struct TranslateJson {
    image: String,
    regions: Vec<RegionOut>,
    ms: u64,
    /// Indices into `regions` whose text came back in the source language.
    ///
    /// Always present, so an empty array means "the server checked and found
    /// none" rather than "this server does not report it". Without it those
    /// regions are indistinguishable from ones OCR never read: they carry a
    /// translation like any other, holding the Japanese verbatim.
    untranslated: Vec<usize>,
    /// Indices into `regions` that were read, erased, and lettered with nothing.
    ///
    /// **`untranslated` structurally cannot report these**, and that is not an
    /// oversight in it -- the two describe different failures. The translation
    /// stage writes an unanswered id back holding its *source* text, so a
    /// segment the model skipped renders as Japanese and appears there. It can
    /// never come back empty. An empty translation is always the other thing: an
    /// entry the model positively answered with an empty string, under a grammar
    /// that pins the number of reply entries and nothing about their content.
    ///
    /// **Under the default plan the box is erased regardless**, because the mask
    /// is written by *detection* from the raw boxes before OCR or translation
    /// run. So this is the only field that reports the one failure with no
    /// visible cause at all: a blank patch where a bubble used to be, and no
    /// Japanese left to hint at it.
    ///
    /// That last sentence is a claim about [`Plan::Full`] and not about this
    /// field. Under [`Plan::KeepArt`] there is no inpainting stage, so nothing is
    /// erased and a dropped region leaves the original artwork -- Japanese
    /// included -- untouched. The region was still read and still lettered with
    /// nothing, which is what this reports; how bad that looks is the plan's
    /// business. Anything phrasing this for a reader has to be true of both, and
    /// the extension's wording is.
    ///
    /// Measured over stored test runs: 59 of 4,689 Japanese regions, on 22
    /// of 426 pages, worst page 13 of 15. On every one of those pages
    /// `untranslated` was empty, `truncated` false and both id counters zero.
    ///
    /// Always present, so an empty array means the server checked. See
    /// [`crate::regions::dropped_indices`] for what it deliberately excludes.
    dropped: Vec<usize>,
    /// Regions whose translation stops mid-sentence with trailing whitespace --
    /// the tell of a reply segment split across two ids, after which every
    /// later region on the page inherits its predecessor's sentence.
    ///
    /// A DIAGNOSTIC, not a gate: it says "look at this page", never "these
    /// regions are wrong" -- one page on record carries two fragments and
    /// recovers alignment before its end. Every other channel on this body
    /// reads clean on the pages this class shifts, which is why it exists; the
    /// mechanism, the census (22 regions, 17 of 213 pages, six shifted) and
    /// the exclusions are on [`crate::regions::split_fragment_indices`].
    /// Always present, so empty means the server checked.
    split_fragments: Vec<usize>,
    /// Indices into `regions` whose placed ink left the reader's own placement
    /// box -- the loud arm of the placement's one honest limit:
    /// the size search never goes below the 9px floor, so a too-small box
    /// SPILLS rather than shrinking into illegibility, and the editor tells
    /// the reader to draw a bigger one. Always present, so an empty array
    /// means "the server checked" -- the same discipline as `dropped`, whose
    /// doc records the mutation check that once proved why a stored field here goes
    /// quietly wrong.
    placement_overflow: Vec<usize>,
    /// Placements whose box left the page raster and was parked back onto it
    /// -- size preserved, flush at the edge, shrunk only if the
    /// box was larger than the page on that axis.
    ///
    /// **These are indices into the caller's own `regions_place` array, not
    /// into `regions`** -- the one field on this body that is, and the
    /// difference is structural rather than a choice: the clamp runs before
    /// the pipeline, so no region exists yet to index. Its neighbour
    /// `placement_overflow` above indexes `regions` and always will.
    ///
    /// The two report opposite halves of the same silence and neither can see
    /// the other's. `placement_overflow` asks whether the placed ink stayed
    /// inside the box; this asks whether the box was on the page. The measured
    /// case that opened it -- an 849x1198 page, a box at `y=1159 height=318`
    /// -- rendered six lines of dialogue off the canvas with `placed=1` and
    /// `placement_overflow: []`, because the ink was perfectly inside a box
    /// that was 279 px below the paper.
    ///
    /// **Hardening, not a user-facing bug**: `boxeditMoveRect` clamps the same
    /// way client-side, so today this can only fire for a caller that is not
    /// this extension. Always present, so an empty array means the server
    /// checked -- the same discipline as `dropped`.
    placement_clamped: Vec<usize>,
    /// How many regions an edit apply's pin kept verbatim.
    /// Always present; 0 means no pin matched or none were sent.
    translation_pins_applied: usize,
    /// Text the detector found, the inpainter erased, and nothing ever painted
    /// back -- today that is sound effects, which are dropped between detection
    /// and OCR.
    ///
    /// Separate from `untranslated` on purpose, because the two answer different
    /// questions and only this one is about the page. `untranslated` is a
    /// *translator* report: a region that reached the translator and came back
    /// in the source language. A region that never became text at all cannot
    /// appear in it, which is why it was empty on every page of a 32-page audit
    /// that included pages with plainly visible untranslated Japanese.
    skipped: Vec<crate::regions::SkippedOut>,
    /// Whether the model stopped on its token cap, which is what turns a dense
    /// page into a partly translated one. Only the local provider can know;
    /// `false` under a remote one means the finish reason was not available.
    truncated: bool,
    /// Reply entries the model addressed to an id it had already answered, whose
    /// text was dropped.
    ///
    /// **The discriminator `untranslated` cannot be.** That list is built from a
    /// "was this id filled in?" mask, so a duplicate appears in it only as its
    /// victim -- the id whose slot was spent -- and reads exactly like a reply
    /// that stopped early. "The model never answered these 7" and "the model
    /// answered 7 of them twice" produce the same `untranslated`, and they want
    /// opposite fixes: a bigger token budget, or none at all.
    ///
    /// Not a hypothetical. The local model decodes under an llguidance grammar
    /// built from the page's JSON schema, and that grammar fixes the *number* of
    /// entries but not their distinctness -- llguidance implements no
    /// `uniqueItems`. "Copy every input ID exactly once" is the one clause of the
    /// prompt with nothing enforcing it, and this is whether it was followed.
    ///
    /// Always present, so `0` means the server checked.
    duplicate_ids: usize,
    /// Reply entries naming an id no submitted region has, likewise dropped.
    ///
    /// Separate from `duplicate_ids` because the two accuse different things. The
    /// id is schema-constrained under the local provider, so a non-zero count
    /// there is evidence about the grammar rather than about the model; under a
    /// remote provider decoding unconstrained it is an ordinary slip.
    out_of_range_ids: usize,
    /// Segments whose first translation arrived visibly cut mid-sentence.
    /// Until this field the repair's whole verdict lived in tracing, so "the
    /// retry fired and repaired it" and "the retry never fired" were
    /// indistinguishable to a reader -- a report emitted and then thrown away
    /// before it reached the browser.
    ///
    /// Always present, so `0` means the translator looked and found none.
    cut_found: usize,
    /// Segments still visibly cut in what SHIPS, after any repair -- recounted
    /// from the final text, so a failed retry is included. `cut_found > 0`
    /// with `still_cut == 0` is a page that was cut and repaired; a non-zero
    /// `still_cut` is a bubble ending mid-word on the page in front of the
    /// reader. Rides the JSON body only -- the PNG path's miss headers are
    /// deliberately untouched, the extension being this field's one consumer.
    still_cut: usize,
    /// What each stage cost, in the order they finished.
    ///
    /// Always present, so an empty array means the run reported nothing rather
    /// than the field being unsupported. A benchmark needs this to attribute a
    /// slow page to the LLM or to OCR; `ms` alone cannot.
    stages: Vec<StageTiming>,
    /// The stages this run actually asked the pipeline for, as the pipeline
    /// itself resolved them -- not as the request spelled them.
    ///
    /// This is `stages` answering a different question. That one is a cost
    /// breakdown and is built from `Progress::Finished`, so a stage that ran but
    /// changed nothing reports `Skipped` and never appears; it cannot be used to
    /// tell whether a stage was *asked for*. A caller sending `skip_inpainting`
    /// has no other way to find out whether the server understood it, because an
    /// unknown multipart field is consumed in silence by design.
    stages_selected: Vec<String>,
    /// Layers the renderer could not set properly -- text that overflowed its
    /// box, or that auto-fit drove down to the readable-size floor.
    ///
    /// Always present, empty meaning the renderer was happy. The renderer has
    /// always computed these and `render_png` used to discard them, which is why
    /// "does the text look wrong on this page?" could only ever be answered by
    /// looking. It is the same shape as the untranslated report: the pipeline
    /// knows, and BireLate was throwing it away.
    layout_warnings: Vec<crate::render::LayoutWarning>,
    /// Every text layer as the renderer actually set it -- solved size, box and
    /// character count.
    ///
    /// `regions[].font_size` is the size detection *authored* from the Japanese
    /// em; auto-fit re-solves it against the box and the renderer ignores the
    /// authored value for bubble text outright. Reading the authored figure and
    /// calling it the rendered one is how a whole rendering change came to
    /// measure as a no-op.
    rendered_text: Vec<crate::render::RenderedText>,
    /// What the server resolved the SOURCE script to for this request.
    ///
    /// `ja` / `ko` / `zh` / `unknown`. **Not the same string as
    /// `regions[].source_language`**: that field echoes the declared language
    /// as a full BCP-47 tag (`zh-CN` beside this field's `zh`), and in output
    /// from older builds it was `ocr.rs`'s hardcoded `ja-JP` fallback on every
    /// region of every source, Chinese and Korean pages included. Reading the
    /// region field as the declared language off old output is how an auditor
    /// concludes a perfectly good arm was run in the wrong language.
    ///
    /// Always present, including `unknown`, so a missing key means "this server
    /// does not report it" rather than "nothing was declared".
    source_script: &'static str,
    /// Text the detector found against exactly one edge of this page and
    /// **deliberately did not admit as a region**, because it scored under the
    /// checkpoint's own `text` floor.
    ///
    /// For a caller assembling a page out of slices, and for nothing else. A
    /// display column running down a webtoon can be plainly present in a slice
    /// and still score 0.2363 against a 0.25 floor -- measured, on one slice of
    /// a test chapter, where missing it cost the seam a whole glyph of a name. So the
    /// joiner needs to know the ink is there.
    ///
    /// **These are not regions and must never be treated as ones.** Nothing on
    /// this page was read, erased, lettered or rendered for them, and they carry
    /// no index into `regions` because they have none: `mask_includes` in the
    /// detection stage builds the erase mask from the raw detections, so
    /// admitting a sub-floor box would strip the artwork under it on every
    /// ordinary page read while nothing ever letters it. The pipeline refuses a
    /// fragment in two places on purpose so that pairing cannot come apart.
    ///
    /// Always present, so an empty array means the server checked and found
    /// none, never "this server does not report it" -- the same rule
    /// `untranslated` states above, for the same reason. The one caveat it
    /// shares with `untranslated`: a run whose detection stage did not execute
    /// reports `[]` too, because there was nothing to check with.
    edge_hints: Vec<EdgeHintOut>,
    /// Prior story pairs this page's prompt was given, or `null` when no story
    /// ran.
    ///
    /// The same number `x-birelate-story-context` carries, because that header
    /// rides on the PNG response only -- so a `format=json` caller had no way to
    /// tell a story that applied from one the server silently ignored. Found by
    /// exactly that: an A/B over a chapter reported `carried=-` on every page
    /// while the translations were in fact changing.
    story_context: Option<usize>,
    /// Glossary terms the prompt carried, `null` when no story ran, `0` when a
    /// story ran with no glossary installed. On the wire for the same reason as
    /// `story_context`: an arm's stored JSON must say which channel it ran
    /// with, or a census counts defects from an off-configuration corpus
    /// without knowing it.
    glossary_terms: Option<usize>,
    /// Whether this page's prompt carried the containment sentence
    /// (`--containment-clause`). On the wire under the same
    /// self-description contract as the two fields above: a prompt-changing
    /// arm whose stored JSON does not say so is an off-configuration corpus
    /// waiting to be counted.
    containment_clause: bool,
}

/// One [`TranslateJson::edge_hints`] entry.
///
/// `(x, y, width, height)` to match `regions[]`, so a caller reads a hint's box
/// the same way it reads a region's, plus which edge it is against. `edge` is a
/// string rather than a boolean because "top" and "bottom" are the two the
/// pipeline can report today and a third is imaginable; a `bottom: bool` would
/// have to be replaced rather than extended.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct EdgeHintOut {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    /// `"top"` or `"bottom"`, straight off `koharu_pipeline::PageEdge`.
    edge: String,
    /// What the detector scored the box, verbatim.
    ///
    /// It is the field that makes a refused box **auditable**. `regions[]` carries
    /// `detection_confidence`; until this existed a hint carried no score at all,
    /// so the sub-floor numbers a seam diagnosis rests on could not be
    /// re-derived from the pipeline's own output.
    ///
    /// **Additive, so no reader breaks.** A client that does not know the field
    /// ignores it, and one that does can now tell 0.2295 from 0.0102 instead of
    /// being told only that *something* was refused. `seam.js` reads a hint through
    /// `seamEffectiveBox`, which takes four named numbers and ignores the rest.
    score: f32,
    /// Whether this box's own ink was walked and found to cross the slice end to
    /// end -- in which case the four numbers above are the **grown** box.
    ///
    /// It is the one thing that lets a REFUSED box declare a slice crossed, and it
    /// is deliberately the server's judgement rather than the browser's: the walk
    /// reads pixels and the browser never sees any. `false` is the original
    /// behaviour unchanged, so the rule that a sub-floor fragment must not reach
    /// `seamSpansSlice` still holds for every hint that is not ink-verified.
    spans: bool,
}

impl From<koharu_pipeline::EdgeHint> for EdgeHintOut {
    fn from(hint: koharu_pipeline::EdgeHint) -> Self {
        Self {
            x: hint.x,
            y: hint.y,
            width: hint.width,
            height: hint.height,
            edge: hint.edge.as_str().to_owned(),
            score: hint.score,
            spans: hint.spans,
        }
    }
}

impl TranslateJson {
    /// The whole body, built from a finished [`Outcome`] in one place.
    ///
    /// A constructor rather than a literal at the call site so that every field
    /// crossing from the run to the wire crosses somewhere a test can stand. The
    /// one that needed it is `dropped`: a mutation check cut its old initialiser to
    /// `Vec::new()` and the entire suite stayed green while every response
    /// shipped an empty list forever. `a_json_body_carries_the_drops_it_found`
    /// stands here now, and it goes red for that cut and for any other field
    /// silently detached from the run that produced it.
    fn of(outcome: Outcome, ms: u64) -> Self {
        // Read before `regions` is moved into the literal, and read off the
        // outcome rather than passed in -- see `Outcome::dropped`.
        let dropped = outcome.dropped();
        let split_fragments = crate::regions::split_fragment_indices(&outcome.regions);
        Self {
            image: format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(&outcome.png)
            ),
            regions: outcome.regions,
            ms,
            untranslated: outcome.untranslated,
            dropped,
            split_fragments,
            placement_overflow: outcome.placement_overflow,
            placement_clamped: outcome.placement_clamped,
            translation_pins_applied: outcome.translation_pins_applied,
            skipped: outcome.skipped,
            truncated: outcome.misses.truncated,
            duplicate_ids: outcome.misses.duplicate_ids,
            out_of_range_ids: outcome.misses.out_of_range_ids,
            cut_found: outcome.misses.cut_found,
            still_cut: outcome.misses.still_cut,
            stages: outcome.stages,
            stages_selected: outcome.selected,
            layout_warnings: outcome.warnings,
            rendered_text: outcome.rendered_text,
            story_context: outcome.story_pairs,
            glossary_terms: outcome.glossary_terms,
            containment_clause: outcome.containment_clause,
            source_script: outcome.source_script.tag(),
            edge_hints: outcome.edge_hints,
        }
    }
}

/// What `/shutdown` answers with.
///
/// A body rather than 204, so the popup can tell "the server accepted and is
/// stopping" from "the fetch failed because it had already gone".
#[derive(Serialize)]
struct ShutdownJson {
    stopping: bool,
}

/// What `/story` answers with after forgetting a story.
#[derive(Serialize)]
struct StoryJson {
    /// Whether anything was actually there. `false` is the ordinary answer for
    /// the first "New story" of a session, not a failure.
    cleared: bool,
}

/// What `/glossary` answers with after installing (or clearing) a glossary.
#[derive(Serialize)]
struct GlossaryJson {
    /// How many terms the series now carries. `0` after a clear.
    terms: usize,
}

/// Everything `/status` tells the popup, and the body `/unload` answers with.
#[derive(Serialize)]
struct StatusJson {
    models_loaded: bool,
    /// Absent rather than zero whenever no unload is pending -- a disabled
    /// budget, or a server holding nothing -- because the popup renders a
    /// countdown from it and a zero would read as "about to free".
    unload_in_secs: Option<u64>,
    /// 0 when `--idle-unload-secs 0` turned the feature off.
    idle_unload_secs: u64,
    vram: Option<VramJson>,
    sufficient: bool,
    needed_bytes: u64,
    provider: &'static str,
    model: String,
    busy: bool,
    /// `null` unless `--hunyuan-substitute` is set. The popup reads this shape.
    ocr_substitute: Option<OcrSubstituteJson>,
}

#[derive(Serialize)]
struct OcrSubstituteJson {
    requested: &'static str,
    served_by: &'static str,
}

/// `budget_bytes`, never `total_bytes`. On Windows these come from DXGI and are
/// *per process*: the budget is what the OS is currently willing to give this
/// process, and neither Ollama's share of the card nor the browser's appears in
/// it. Presented as the card's capacity it would be wrong by tens of gigabytes.
#[derive(Serialize)]
struct VramJson {
    budget_bytes: u64,
    available_bytes: u64,
    device: String,
}

/// The four token-gated handlers, so the route table can be built with stubs in
/// place of anything that needs a pipeline behind it.
pub struct Handlers {
    pub translate: MethodRouter,
    pub status: MethodRouter,
    pub warmup: MethodRouter,
    pub unload: MethodRouter,
    pub shutdown: MethodRouter,
    pub story: MethodRouter,
    pub glossary: MethodRouter,
}

pub fn app(state: AppState) -> Router {
    let guard = state.guard.clone();
    let max_upload_bytes = state.max_upload_bytes;
    router(
        guard,
        Handlers {
            translate: post(translate).with_state(state.clone()),
            status: get(status).with_state(state.clone()),
            warmup: post(warmup).with_state(state.clone()),
            unload: post(unload).with_state(state.clone()),
            shutdown: post(shutdown).with_state(state.clone()),
            story: post(clear_story).with_state(state.clone()),
            glossary: post(set_glossary).with_state(state),
        },
        max_upload_bytes,
    )
}

/// The route table and every guard layer.
pub fn router(guard: GuardState, handlers: Handlers, max_upload_bytes: usize) -> Router {
    // `route_layer`, not `layer`: neither the token check nor the upload cap
    // should turn a wrong-method request into a 401. Cloned into the closure so
    // the guard itself is still free to move into the outer layer below.
    let token = guard.clone();
    let gated = move |handler: MethodRouter| {
        handler.route_layer(middleware::from_fn_with_state(token.clone(), require_token))
    };
    Router::new()
        .route(
            "/translate",
            // axum's own default is 2 MB, which a real manga page exceeds.
            gated(handlers.translate).route_layer(DefaultBodyLimit::max(max_upload_bytes)),
        )
        .route("/status", gated(handlers.status))
        .route("/warmup", gated(handlers.warmup))
        .route("/unload", gated(handlers.unload))
        .route("/shutdown", gated(handlers.shutdown))
        .route("/story", gated(handlers.story))
        .route("/glossary", gated(handlers.glossary))
        // The one route not passed through `gated`. See the module comment.
        .route("/health", get(health))
        .layer(middleware::from_fn_with_state(guard, guard_origin_host))
        .layer(TraceLayer::new_for_http())
        // Outermost, so it covers the guards and the trace layer as well as the
        // handlers.
        .layer(middleware::from_fn(catch_panics))
    // No CorsLayer, and no Access-Control-Allow-Origin, ever. With no OPTIONS
    // route, a page that tries to send X-Koharu-Token preflights and fails
    // closed, so it can never read a response even if one were to leak.
}

/// Turns a panic below this layer into a plain-text 500.
///
/// Without it a panic in a handler frame unwinds into the connection task axum
/// spawned for that request. tokio catches it and drops the task, so the socket
/// closes with no response on it and the extension's `fetch` fails with a network
/// error -- a connection reset standing in for a server fault it could have
/// displayed. `/unload` is the concrete risk: it calls koharu's model destructors
/// on the handler's own stack.
///
/// The server itself is unaffected either way; only this request's connection
/// was ever at stake, which is why `/health` and `/status` keep answering.
async fn catch_panics(request: axum::extract::Request, next: middleware::Next) -> Response {
    match AssertUnwindSafe(next.run(request)).catch_unwind().await {
        Ok(response) => response,
        Err(payload) => panicked("the server", &*payload).into_response(),
    }
}

/// A panic is a 500 like any other failure, carrying the panic's own message.
///
/// `what` leads because the payload alone does not say which part of a request
/// died, and the extension shows this text with no other context.
fn panicked(what: &str, payload: &(dyn Any + Send)) -> ApiError {
    let text = panic_text(payload);
    tracing::error!(panic = %text, "{} panicked", what);
    ApiError::internal(clip(&format!("{what} panicked: {text}"), CLIP_BUDGET))
}

/// A task that did not return, whether it panicked or was cancelled.
///
/// The panic's message is reflected rather than dropped: this is the path a stage
/// panic that koharu did not catch arrives on, and a generic "did not finish"
/// there is what let a cuDNN fault look like anything at all. Cancellation stays
/// generic because the only thing that cancels these is the runtime shutting
/// down, which the caller is in no position to act on.
fn task_failed(what: &'static str, error: tokio::task::JoinError) -> ApiError {
    if error.is_cancelled() {
        return ApiError::internal(RENDER_TASK_FAILED);
    }
    panicked(what, &*error.into_panic())
}

/// Constant body: nothing here should identify the build or leak the token.
async fn health() -> &'static str {
    "ok"
}

async fn status(State(state): State<AppState>) -> Json<StatusJson> {
    Json(status_of(&state))
}

/// Starts a load in the background and answers at once. Loading four models
/// takes far longer than the extension will wait; the popup polls `/status`.
async fn warmup(State(state): State<AppState>) -> Result<Response, ApiError> {
    // Answered before the headroom is measured, and that order matters: a server
    // that has already loaded its models has already spent the room a cold start
    // needs, so asking whether there is room for one would refuse the very state
    // this endpoint exists to reach.
    if models_loaded(&state) {
        return Ok((StatusCode::ACCEPTED, "already loaded\n").into_response());
    }
    // Always the cold threshold: warming up is by definition about weights that
    // are not resident yet.
    let headroom = vram::headroom(&state.vram, false, state.cold_reserve_bytes);
    if !headroom.sufficient {
        return Err(insufficient_vram(&headroom));
    }
    if !state.claim_warmup() {
        return Ok((StatusCode::ACCEPTED, "already loading\n").into_response());
    }
    tokio::spawn(warm_models(state));
    Ok((StatusCode::ACCEPTED, "loading\n").into_response())
}

async fn unload(State(state): State<AppState>) -> Result<Json<StatusJson>, ApiError> {
    {
        // Never waits for the gate. A page, or a warmup, can take minutes and
        // the popup's button has to answer now; "busy" is a truthful answer, and
        // the idle budget frees those weights anyway once the work finishes.
        let Ok(_gpu) = state.gate.try_lock() else {
            return Err(ApiError::new(StatusCode::SERVICE_UNAVAILABLE, UNLOAD_BUSY));
        };
        let freed = free_models(&state.pipeline);
        state.idle.cleared();
        tracing::info!(freed, "freed the models on request");
    }
    Ok(Json(status_of(&state)))
}

/// Ends the process, for the popup's Stop button.
///
/// Deliberately does NOT wait for the gate. A page in flight can take minutes,
/// and axum's graceful shutdown already lets it finish and answer -- taking the
/// gate here would only make Stop hang for exactly as long as the work the user
/// is trying to stop. It answers before the listener closes, so the popup gets a
/// 200 rather than a connection reset.
///
/// Token-gated like every other mutating route. Unauthenticated it would let any
/// page that can reach loopback kill the reader's server.
async fn shutdown(State(state): State<AppState>) -> Json<ShutdownJson> {
    tracing::info!("shutdown requested over HTTP");
    state.shutdown.notify_waiters();
    Json(ShutdownJson { stopping: true })
}

/// Forgets a story's accumulated context, for the popup's "New story" button.
///
/// Takes the id to forget as a `story` multipart field. Clearing by id rather
/// than clearing everything means a second tab reading a different story is not
/// collateral -- and it is why the popup mints a fresh id as well as calling
/// this: the reset is belt and braces, since an unknown id carries nothing
/// anyway.
async fn clear_story(
    State(state): State<AppState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<StoryJson>, ApiError> {
    let multipart = multipart.map_err(|rejection| rejection_error(&rejection))?;
    let form = read_form(multipart, state.max_upload_bytes).await?;
    let Some(id) = form.story.as_deref().map(str::trim).filter(|id| !id.is_empty()) else {
        return Err(ApiError::bad_request("story is required"));
    };
    if !crate::story::valid_id(id) {
        return Err(ApiError::bad_request(
            "story must be 1-64 characters of A-Z, a-z, 0-9, '-' or '_'",
        ));
    }
    let cleared = state.stories.clear(id);
    tracing::info!(story = id, cleared, "story context cleared");
    Ok(Json(StoryJson { cleared }))
}

/// Installs, replaces or clears the per-series glossary for a story id.
///
/// Takes the id as a `story` multipart field, exactly like `/story`, and the
/// terms as a `terms` field holding a JSON array of `{source, translation}`.
/// An absent or empty `terms` clears. Replacement is whole-glossary on
/// purpose — a partial update could not be told apart from a stale one.
async fn set_glossary(
    State(state): State<AppState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<GlossaryJson>, ApiError> {
    let multipart = multipart.map_err(|rejection| rejection_error(&rejection))?;
    let form = read_form(multipart, state.max_upload_bytes).await?;
    let Some(id) = form.story.as_deref().map(str::trim).filter(|id| !id.is_empty()) else {
        return Err(ApiError::bad_request("story is required"));
    };
    if !crate::story::valid_id(id) {
        return Err(ApiError::bad_request(
            "story must be 1-64 characters of A-Z, a-z, 0-9, '-' or '_'",
        ));
    }
    let terms = match form.terms.as_deref().map(str::trim).filter(|terms| !terms.is_empty()) {
        None => Vec::new(),
        Some(json) => crate::glossary::parse_terms(json).map_err(ApiError::bad_request)?,
    };
    let stored = state.glossaries.set(id, terms);
    tracing::info!(story = id, terms = stored, "glossary installed");
    Ok(Json(GlossaryJson { terms: stored }))
}

fn status_of(state: &AppState) -> StatusJson {
    let loaded = models_loaded(state);
    let headroom = vram::headroom(&state.vram, loaded, state.cold_reserve_bytes);
    // `try_lock` rather than waiting for the gate: a status request has to
    // answer while it is held, since that is exactly when the popup asks.
    let busy = state.gate.try_lock().is_err();
    status_json(&state.pinned, &state.defaults, &state.idle, &headroom, loaded, busy)
}

/// The `/status` body from `status_of`'s live readings. Split out so a test
/// reaches this construction too: `AppState` owns a `Pipeline`, and no unit
/// test can build one.
fn status_json(
    pinned: &Pinned,
    defaults: &Defaults,
    idle: &crate::idle::IdleClock,
    headroom: &vram::Headroom,
    loaded: bool,
    busy: bool,
) -> StatusJson {
    StatusJson {
        models_loaded: loaded,
        unload_in_secs: idle.remaining().map(|left| left.as_secs()),
        idle_unload_secs: idle.budget().map_or(0, |budget| budget.as_secs()),
        vram: vram_json(&headroom.vram),
        sufficient: headroom.sufficient,
        needed_bytes: headroom.needed_bytes,
        provider: pinned.wire_provider,
        model: pinned.llm.clone(),
        busy,
        ocr_substitute: defaults.ocr_substitute.as_ref().map(|engine| OcrSubstituteJson {
            requested: "hunyuan-ocr-1.5",
            served_by: models::ocr_model_name(engine),
        }),
    }
}

/// `null` rather than zeroes when there is no reading. An empty device list is
/// what a monitor that has not published yet, and a `--cpu` server, both look
/// like; rendering either as "0 bytes free" would be a lie the popup repeats.
fn vram_json(vram: &Vram) -> Option<VramJson> {
    match vram {
        Vram::Unknown => None,
        Vram::Known {
            device,
            budget_bytes,
            available_bytes,
        } => Some(VramJson {
            budget_bytes: *budget_bytes,
            available_bytes: *available_bytes,
            device: device.clone(),
        }),
    }
}

/// The 507 for a card that cannot take the run.
///
/// koharu itself would never refuse -- its admission control only chooses what
/// to evict -- but on a card without room "start anyway" means a native abort with
/// exit code 3, no Rust panic and no output at all. A refusal the user can read
/// beats that. The shortfall leads, because it is the number they can act on.
fn insufficient_vram(headroom: &vram::Headroom) -> ApiError {
    ApiError::new(
        StatusCode::INSUFFICIENT_STORAGE,
        clip(
            &format!(
                "insufficient VRAM: {:.1} GB short of the {:.1} GB a cold start needs; free some, \
                 or POST /unload",
                vram::gb(vram::shortfall(headroom)),
                vram::gb(headroom.needed_bytes)
            ),
            CLIP_BUDGET,
        ),
    )
}

/// The 507 for a model change the card cannot afford to make.
///
/// Separate from `insufficient_vram` because both of its sentences would be
/// wrong here. Nothing is short *now* -- the card is warm and working -- and
/// `POST /unload` is no advice at all, since the reload frees those same weights
/// by itself. What the reader can act on is the change they just made, so that
/// is what the message names.
fn insufficient_vram_for_reload(headroom: &vram::Headroom) -> ApiError {
    ApiError::new(
        StatusCode::INSUFFICIENT_STORAGE,
        clip(
            &format!(
                "insufficient VRAM: changing the models forces a cold start, {:.1} GB short of \
                 the {:.1} GB it needs",
                vram::gb(vram::shortfall(headroom)),
                vram::gb(headroom.needed_bytes)
            ),
            CLIP_BUDGET,
        ),
    )
}

async fn translate(
    State(state): State<AppState>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    let started = Instant::now();
    let multipart = multipart.map_err(|rejection| rejection_error(&rejection))?;
    let form = read_form(multipart, state.max_upload_bytes).await?;
    let selection = selection_from(&form, &state.defaults, &state.pinned)?;
    /* Deliberately not a field of `Selection`. Stage choice lives on koharu's
     * `Request`, not on `PipelineConfig`, and `reconcile` reloads the whole
     * pipeline on any `applied != desired` -- so hanging this off the selection
     * would rebuild the `StageRunner`, and with it the `Residency`, every time
     * the reader flipped the switch. That throws away the learned per-stage
     * memory profiles, which `Pipeline::unload_models` exists to preserve.
     * Carried alongside instead, where it changes the operation and nothing
     * else, and toggling it costs no reload at all. */
    let skip_inpainting = match form.skip_inpainting.as_deref() {
        Some(value) => models::parse_flag("skip_inpainting", value)?,
        None => false,
    };
    let clean_only = match form.clean_only.as_deref() {
        Some(value) => models::parse_flag("clean_only", value)?,
        None => false,
    };
    /* Parsed by the same table as every other flag, so a malformed value is a
     * 400 rather than a silent `false`. Silence here would be the worst of both:
     * the seam would look wired, the join would still be refused, and nothing
     * would say why -- which is the exact failure this change exists to remove. */
    let joined_page = match form.joined.as_deref() {
        Some(value) => models::parse_flag("joined", value)?,
        None => false,
    };
    /* Refused rather than skimmed: a half-parseable list would place the cut
     * somewhere the caller did not say, and the rule reading it decides what
     * gets MINTED onto the page. Absent is fine -- the rule then never fires --
     * garbled is not. */
    let joined_boundaries: Vec<f64> = match form.joined_boundaries.as_deref() {
        Some(value) => value
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(|part| {
                part.parse::<f64>()
                    .ok()
                    .filter(|offset| offset.is_finite() && *offset > 0.0)
                    .ok_or_else(|| {
                        ApiError::bad_request(
                            "joined_boundaries must be comma-separated positive pixel offsets",
                        )
                    })
            })
            .collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    let detect_only = match form.detect_only.as_deref() {
        Some(value) => models::parse_flag("detect_only", value)?,
        None => false,
    };
    /* The retry button's re-roll knob, refused-on-garbage like every declared
     * field here: a garbled seed reading as "absent" would keep the fixed
     * constant and hand back the identical page -- a retry button that
     * silently does nothing. Absent is every non-retry caller. */
    let translation_seed = parse_seed(form.seed.as_deref())?;
    /* The box editor's two lists, refused-on-garbage for
     * `joined_boundaries`' reason, sharpened: a half-parsed removal would
     * leave a box the reader deleted ERASING ITS ARTWORK, and a half-parsed
     * addition would silently drop the correction they drew. */
    let added_regions = parse_caller_regions("regions_add", form.regions_add.as_deref())?;
    let removed_regions = parse_caller_regions("regions_remove", form.regions_remove.as_deref())?;
    /* The editor's placement overrides, same discipline: a
     * half-parsed placement would letter English somewhere the reader did
     * not choose, and nothing on the wire would say so. */
    let placements = parse_caller_placements(form.regions_place.as_deref())?;
    /* The edit apply's wording pins, same discipline again: a
     * skimmed pin re-rolls a bubble the reader thought they were keeping. */
    let pins = parse_translation_pins(form.translation_pins.as_deref())?;
    /* The profile axis, refused-on-garbage like every declared field
     * here: an ignored declaration is the quiet failure the axis exists to
     * remove. INERT past this point -- validated and logged, no behavioral
     * consumer yet; `models::PageProfile` says what the change that wires one
     * must also do. Absent is every caller that predates the field. */
    let page_profile = match form.profile.as_deref() {
        Some(value) => Some(models::parse_profile(value)?),
        None => None,
    };
    if let Some(profile) = page_profile {
        tracing::debug!(?profile, "caller declared a page profile");
    }
    // Contradictory rather than merely redundant: one asks for everything but the
    // eraser, one for the eraser and nothing else, one for neither. Refusing is
    // the only reading that cannot silently do the opposite of what was meant.
    // The choice lives in `choose_plan` so it can be tested without a request.
    let plan = choose_plan(skip_inpainting, clean_only, detect_only)?;
    /* Validated at the edge: the id becomes a HashMap key and a response header
     * value, and a story with no dialogue in it yet is indistinguishable from a
     * typo unless the malformed one is refused outright. CleanOnly carries no
     * text at all, so a story cannot be built from it. */
    let story = match form.story.as_deref().map(str::trim).filter(|id| !id.is_empty()) {
        Some(id) if !crate::story::valid_id(id) => {
            return Err(ApiError::bad_request(
                "story must be 1-64 characters of A-Z, a-z, 0-9, '-' or '_'",
            ));
        }
        Some(id) if plan.letters_text() => Some(id.to_owned()),
        _ => None,
    };

    let want_json = wants_json(
        form.format.as_deref(),
        headers
            .get(header::ACCEPT)
            .and_then(|value| value.to_str().ok()),
    );
    let image = form
        .image
        .ok_or_else(|| ApiError::bad_request(MISSING_IMAGE))?;

    // Decoding and the in-memory session are blocking and CPU-only, so they
    // happen before the gate rather than queued behind someone else's page.
    let prepared = tokio::task::spawn_blocking(move || prepare_page(&image))
        .await
        .map_err(|error| task_failed("decoding the page", error))??;

    /* Read back off the selection rather than re-derived. It is resolved in
     * `selection_from`, which is the one place that can see both what the request
     * declared and which engine it asked for -- and, since the erase veto began
     * needing it too, the only place early enough to reach `PipelineConfig`.
     * A second resolution site is a second place for the two to disagree about
     * what language the page is in. */
    let script = selection.source_script;

    // Spawned rather than awaited inline: a closed tab aborts the fetch, which
    // would otherwise drop this future in the middle of a model run.
    let outcome = tokio::spawn(translate_page(
        state,
        prepared,
        selection,
        want_json,
        plan,
        story,
        script,
        joined_page,
        joined_boundaries,
        translation_seed,
        added_regions,
        removed_regions,
        placements,
        pins,
    ))
    .await
    .map_err(|error| task_failed("the translation", error))??;

    if want_json {
        let payload = TranslateJson::of(outcome, started.elapsed().as_millis() as u64);
        return Ok((StatusCode::OK, Json(payload)).into_response());
    }

    // The content type must be set explicitly: `Bytes` responds as
    // application/octet-stream, and the extension turns the body into a data:
    // URL that it assigns to an <img>, which would then render nothing.
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    miss_headers(&mut headers, &outcome);
    /* Sent on every page that reported a stage list, clean or not -- unlike the
     * two above, which are silent when there is nothing to say. Absence is the
     * signal here. An unknown multipart field is consumed in silence (see
     * `read_form`), so a client that sends `skip_inpainting` to a server built
     * before this existed gets a fully inpainted page and a 200, with nothing in
     * the body to say so -- and the extension would file that erased artwork in
     * its 24-hour cache under a keep-the-art fingerprint. A server binary older
     * than the extension is the ordinary state here, since web-ext reloads the
     * extension on every save and the server is rebuilt by hand.
     *
     * Skipped when the list is empty rather than sent blank, because an empty
     * value is not "no inpainting" -- it is the pipeline having reported nothing
     * -- and a reader splitting it on commas would find no "inpainting" in it and
     * conclude the art survived. Unknown must be indistinguishable from absent. */
    if !outcome.selected.is_empty()
        && let Ok(value) = HeaderValue::from_str(&outcome.selected.join(","))
    {
        headers.insert("x-birelate-stages", value);
    }
    /* Same reasoning as `x-birelate-stages`, for the same reason it exists: an
     * unknown multipart field is consumed in silence, so a `story` sent to a
     * server built before stories produces a context-free translation with a 200
     * and nothing to distinguish it. The value is the number of prior pairs this
     * page was actually given, so "0" on page five says the story was dropped
     * while the header being absent says the server has no idea what a story is. */
    if let Some(pairs) = outcome.story_pairs
        && let Ok(value) = HeaderValue::from_str(&pairs.to_string())
    {
        headers.insert("x-birelate-story-context", value);
    }
    /* A count, not the detail: the PNG path has no region list to index into,
     * and the extension shows this in a narrow window. "2 of 9 layers did not
     * fit" is the whole of what a reader can act on -- the per-layer numbers are
     * for `format=json`. Same rule as the miss report: silent on a clean page,
     * so an ordinary render answers header for header as it always has. */
    if !outcome.warnings.is_empty()
        && let Ok(value) = HeaderValue::from_str(&outcome.warnings.len().to_string())
    {
        headers.insert("x-birelate-layout-warnings", value);
    }
    Ok((StatusCode::OK, headers, Bytes::from(outcome.png)).into_response())
}

/// The PNG path's whole miss report, gate included.
///
/// A function rather than a block inside `translate` so the gate itself is
/// reachable from a test. The gate is the interesting half: `dropped` is checked
/// beside `worth_reporting` and is deliberately NOT folded into it, because
/// `Misses` is the *translator's* report of its own reply and a drop is not in
/// it -- every counter on it reads clean on all 59 measured instances. A page
/// can therefore have nothing to say up there and still be the page worth
/// looking at, and folding the two together would leave the PNG caller with no
/// drop header at all on exactly those pages.
///
/// Silent on a page with nothing to say, so an ordinary render answers byte for
/// byte and header for header as it always has.
fn miss_headers(headers: &mut HeaderMap, outcome: &Outcome) {
    let dropped = outcome.dropped();
    if outcome.misses.worth_reporting() || !dropped.is_empty() {
        untranslated_headers(
            headers,
            &outcome.misses,
            dropped.len(),
            outcome.regions.len(),
        );
    }
}

/// `X-BireLate-Untranslated: <missed>/<total>`, the truncation flag, and — each
/// only when there is one — `X-BireLate-Dropped-Ids: <duplicated>/<out of
/// range>` and `X-BireLate-Dropped: <read but unlettered>/<regions>`.
///
/// One header rather than two numbers because the pair is only meaningful
/// together: "3" alone reads as an error count, "3/41" reads as a mostly good
/// page. Both are ASCII digits and a slash, so `from_str` cannot fail on a
/// value this builds -- but it is still handled rather than unwrapped, because
/// a header that failed to build must not take the page down with it.
fn untranslated_headers(
    headers: &mut HeaderMap,
    misses: &Misses,
    dropped: usize,
    regions: usize,
) {
    if let Ok(value) =
        HeaderValue::from_str(&format!("{}/{}", misses.entities.len(), misses.segments))
    {
        headers.insert("x-birelate-untranslated", value);
    }
    headers.insert(
        "x-birelate-truncated",
        HeaderValue::from_static(if misses.truncated { "true" } else { "false" }),
    );
    /* `<duplicated>/<out of range>`, and only when there is one, because this is
     * the header that stops the pair above being read as an all-clear. A page
     * whose ids were mis-addressed can answer `0/41` and `false` up there and
     * still be the page worth looking at -- and where it does miss ids, the
     * denominator above says how many bubbles are Japanese while this says
     * whether the model ran short or simply lost count. Both are two ASCII
     * integers and a slash, so `from_str` cannot fail on a value built here. */
    if (misses.duplicate_ids > 0 || misses.out_of_range_ids > 0)
        && let Ok(value) = HeaderValue::from_str(&format!(
            "{}/{}",
            misses.duplicate_ids, misses.out_of_range_ids
        ))
    {
        headers.insert("x-birelate-dropped-ids", value);
    }
    /* `<dropped>/<regions on the page>`, and only when there is one. The
     * denominator is deliberately NOT `misses.segments` like the header above:
     * a drop is counted off the region walk, and on every measured instance the
     * translator reported nothing at all, so `segments` is 0 there and would
     * make this read `3/0`. Two adjacent headers with different denominators is
     * the lesser evil -- see `TranslateJson::dropped`, which has the same
     * asymmetry for the same reason.
     *
     * Not to be confused with `x-birelate-dropped-ids` above, which counts reply
     * ENTRIES the translator threw away. This counts BOXES on the page that were
     * erased and painted with nothing. */
    if dropped > 0
        && let Ok(value) = HeaderValue::from_str(&format!("{dropped}/{regions}"))
    {
        headers.insert("x-birelate-dropped", value);
    }
}

/// What the translation stage reported about its own output.
///
/// Default -- no entities, no segments -- is the honest reading of a run that
/// said nothing: the stage only emits when it has a miss to report.
#[derive(Clone, Debug, Default)]
struct Misses {
    entities: Vec<EntityId>,
    segments: usize,
    truncated: bool,
    /// Reply entries addressed to an id already answered. See `TranslateJson`
    /// for why this is not derivable from `entities`.
    duplicate_ids: usize,
    /// Reply entries naming an id no submitted segment has.
    out_of_range_ids: usize,
    /// Segments whose first reply arrived visibly cut mid-sentence. See
    /// `TranslateJson` for the pair's reader-facing meaning.
    cut_found: usize,
    /// Segments still visibly cut in what ships, after any repair.
    still_cut: usize,
}

impl Misses {
    /// Whether the run said anything the caller needs to hear.
    ///
    /// Truncation counts on its own. A reply cut off *inside* the final
    /// segment's text is repaired into a complete entry, so every id is
    /// answered, `entities` is empty -- and that bubble still ends mid-word.
    /// The budget is sized to be just enough, so an overflow cuts near the end
    /// of the array and landing inside the last item is about as likely as
    /// landing before it. Gating on `entities` alone reports half of all
    /// overflows as a clean page.
    ///
    /// A mis-addressed reply counts on its own for the same shape of reason. A
    /// provider free to answer more entries than were asked for can name one id
    /// twice and still cover every one, leaving `entities` empty and `truncated`
    /// clear -- and that page is the only direct evidence there is about the one
    /// prompt clause the grammar does not enforce.
    fn worth_reporting(&self) -> bool {
        !self.entities.is_empty()
            || self.truncated
            || self.duplicate_ids > 0
            || self.out_of_range_ids > 0
    }

    /// A page where every single segment came back in the source language.
    ///
    /// Not a partial result: nothing was translated, so the render is the
    /// original text laid out in a translation font, which is a worse lie than
    /// an error. A page with *some* misses is left alone -- it is readable, and
    /// throwing away the bubbles that did work costs the reader another full
    /// run.
    fn nothing_translated(&self) -> bool {
        self.segments > 0 && self.entities.len() >= self.segments
    }
}

struct Outcome {
    png: Vec<u8>,
    regions: Vec<RegionOut>,
    /// Indices into `regions`, so empty whenever the regions were not read.
    untranslated: Vec<usize>,
    misses: Misses,
    /// Detected regions erased without ever being translated. See
    /// `regions::skipped` for why `untranslated` cannot answer this.
    skipped: Vec<crate::regions::SkippedOut>,
    stages: Vec<StageTiming>,
    /// The resolved stage list, straight off `Progress::Started`.
    selected: Vec<String>,
    /// What the renderer thought of its own output, per layer.
    warnings: Vec<crate::render::LayoutWarning>,
    /// Every text layer as the renderer set it. See `render::Rendered`.
    rendered_text: Vec<crate::render::RenderedText>,
    /// Prior pairs this page's prompt was given, or `None` when no story ran.
    story_pairs: Option<usize>,
    /// Glossary terms this page's prompt was given, or `None` when no story
    /// ran. `Some(0)` is the ordinary answer: a story with no installed
    /// glossary.
    glossary_terms: Option<usize>,
    /// Whether the prompt carried the containment sentence -- read off the
    /// `desired_config` this run actually installed, the same carried-not-
    /// re-derived rule as `source_script` below.
    containment_clause: bool,
    /// The source script the request RESOLVED to, straight off `Selection`.
    ///
    /// Carried on the outcome rather than re-derived at the wire, for the reason
    /// `selection_from` gives: a second resolution site is a second place for the
    /// two to disagree about what language the page is in.
    source_script: crate::labels::SourceScript,
    /// Sub-floor text against a page edge, straight off `Progress::EdgeHints`.
    /// See [`TranslateJson::edge_hints`] -- these are NOT regions and carry no
    /// index into `regions`.
    edge_hints: Vec<EdgeHintOut>,
    /// Indices into `regions` whose placed ink left the reader's own placement
    /// box. Always computed -- empty means "checked, all inside",
    /// never "nobody looked" -- because the placement's one honest limit (the
    /// 9px font floor makes a too-small box spill rather than shrink) is
    /// invisible without it.
    placement_overflow: Vec<usize>,
    /// Indices into the caller's own `regions_place` array whose `place` rect
    /// had to be parked back onto the page.
    ///
    /// **Not indices into `regions`**, unlike its neighbour above -- the clamp
    /// runs at the top of `translate_page`, before the pipeline has settled a
    /// single region, so the only thing it can name is the entry the caller
    /// sent. See [`crate::placement::clamp_to_page`].
    placement_clamped: Vec<usize>,
    /// Regions whose wording an edit apply's pin kept. 0 on every
    /// request that carried no pins; the pins pass logs asked-vs-applied.
    translation_pins_applied: usize,
}

impl Outcome {
    /// Indices into [`Self::regions`] that were read and lettered with nothing.
    /// See [`crate::regions::dropped_indices`]; `untranslated` structurally
    /// cannot report these.
    ///
    /// **Derived rather than stored, and that is the whole design of it.** This
    /// began as a field on the literal below, computed once in `translate_page`
    /// and copied out at the two response sites -- and a mutation check showed that
    /// replacing that one initialiser with `Vec::new()` left the entire suite
    /// green while every response on the wire shipped `dropped: []` forever.
    /// That is *the exact failure this whole report exists to catch*: a channel
    /// that says "checked, found none" while nothing checked anything. A field
    /// that can be assigned can be assigned wrongly; a method that reads the
    /// regions it reports on cannot be cut without a test going red, which
    /// `an_outcome_reports_its_own_drops` is.
    ///
    /// Cheap enough to call per response: a trimmed-emptiness test over a few
    /// dozen strings already in memory, beside a page that just spent seconds on
    /// the device.
    fn dropped(&self) -> Vec<usize> {
        crate::regions::dropped_indices(&self.regions)
    }
}

/// What one stage cost, in the order the scheduler finished them.
///
/// Without this a caller sees only the page total, which cannot answer the
/// question that actually decides model choice: how much of a page is the LLM
/// and how much is OCR. `run.exe` has always printed these; the server was
/// dropping them on the floor.
///
/// The times can sum to more than the page total, and **not because stages
/// overlap.** `StageRunner::run` stamps its clock and only then
/// awaits `Residency::enter`, which blocks on a one-permit semaphore, so a stage
/// that queues behind another bills the queue to itself. `wait_ms` is that
/// queue, split out: `ms - wait_ms` is what the stage actually cost.
///
/// The distinction is not academic. Measured over 205 pages of a test volume,
/// `inpainting` reported 356 ms median against 187 ms on a `clean_only` run of
/// the same pages with no OCR stage in the pipeline -- and `inpainting - ocr`
/// stayed inside a 106-217 ms band while OCR itself ranged 1.0-5.9 s, which is
/// one number tracking another rather than a cost of its own. Inpainting is the
/// smallest device stage, not the second largest, and every structural idea
/// sized against its old figure was sized against roughly 2.4x the truth.
///
/// `ms` keeps its old meaning on purpose, so new output stays comparable with
/// old. Subtract, do not re-read.
#[derive(Debug, Clone, Serialize)]
struct StageTiming {
    stage: String,
    model: String,
    ms: u64,
    /// The part of `ms` spent queued for the accelerator lane. `0` on a stage
    /// that never waited, which is every stage on an idle server.
    wait_ms: u64,
}

/// Everything that touches the device, as one serialized unit.
///
/// This frame owns the two things a failed run must not damage -- the GPU permit
/// and the idle countdown -- and nothing in it reaches native code. That is
/// deliberate: see the `catch_unwind` below.
async fn translate_page(
    state: AppState,
    prepared: PreparedPage,
    selection: Selection,
    want_regions: bool,
    plan: Plan,
    story: Option<String>,
    script: crate::labels::SourceScript,
    // The caller assembled this image so its text ends inside it -- the seam.
    // Travels beside `carried` all the way down because it lands in the same
    // place, on the `Request` rather than the config, for the same reason.
    joined_page: bool,
    joined_boundaries: Vec<f64>,
    // The retry's re-roll. Same carriage as the pair above and for the same
    // reason: a seed on the config would make every re-roll a reload.
    translation_seed: Option<u32>,
    // The box editor's lists, same carriage again.
    added_regions: Vec<koharu_pipeline::CallerRegion>,
    removed_regions: Vec<koharu_pipeline::CallerRegion>,
    // The editor's placement overrides. Owned here rather than by
    // `run_on_device` because the overflow report below re-reads them after
    // the render comes back -- and because the page clamp below has to happen
    // once, above both readers.
    mut placements: Vec<CallerPlacement>,
    // The edit apply's wording pins, applied before the gates.
    pins: Vec<TranslationPin>,
) -> Result<Outcome, ApiError> {
    let PreparedPage { mut session, page } = prepared;
    /* The off-page half. HARDENING, not a live bug: the editor
     * already clamps a dragged box to the page (`boxeditMoveRect`), so this
     * cannot be reached through the UI today. It is here because of what an
     * off-page `place` rect DOES -- measured on an 849x1198 page, a box at
     * `y=1159 height=318` lettered six lines off the canvas while the run
     * reported `placed=1` and `placement_overflow: []`.
     *
     * Deliberately the FIRST thing this function does to the placements, and
     * the only clamp site. Everything downstream reads the clamped list: the
     * frame write inside `finish_scene`, and `overflowed` after the render.
     * Clamping inside the placement pass instead would leave `overflowed`
     * judging the placed ink against the original off-page box and reporting
     * a spill that no longer exists. */
    let placement_clamped =
        crate::placement::clamp_to_page(&session.snapshot(), page, &mut placements);
    let desired = desired_config(&selection, &state.defaults, &state.pinned);

    /* Read ONCE, before the run, and reused for both the prompt and the header.
     * Reading it again afterwards would count the pairs this very page just
     * contributed, and reading it twice would let the two disagree. */
    let carried: std::sync::Arc<[koharu_translator::TranslationContext]> =
        story.as_deref().map_or_else(
            || std::sync::Arc::from([] as [koharu_translator::TranslationContext; 0]),
            |id| state.stories.context(id),
        );
    let carried_pairs = story.as_ref().map(|_| carried.len());
    /* Same read-once rule as `carried`, same key. Series without a glossary --
     * almost all of them -- read an empty slice and cost nothing. */
    let glossary: std::sync::Arc<[koharu_translator::TranslationContext]> =
        story.as_deref().map_or_else(
            || std::sync::Arc::from([] as [koharu_translator::TranslationContext; 0]),
            |id| state.glossaries.terms(id),
        );
    let glossary_terms = story.as_ref().map(|_| glossary.len());

    // Before the gate: a refusal is only useful if it lands before the page has
    // queued behind everything else. A warm server is never refused -- its
    // weights are already paid for, so what a cold start would have needed says
    // nothing about whether this page can run.
    let loaded = models_loaded(&state);
    let headroom = vram::headroom(&state.vram, loaded, state.cold_reserve_bytes);
    if !headroom.sufficient && !loaded {
        return Err(insufficient_vram(&headroom));
    }

    /* Armed from here, so that however the run ends the countdown restarts at
     * the end of it. Measured from the start it could fire while the page it was
     * meant to outlast is still going, and a run that failed part way through
     * still leaves the stages before the failure resident.
     *
     * Named, not `_idle`, because the success path drops it by hand at the gate
     * rather than at the end of the function. See the `drop` below. */
    let idle_guard = TouchOnDrop(&state.idle);

    let mut gpu = tokio::time::timeout(state.queue_timeout, state.gate.clone().lock_owned())
        .await
        .map_err(|_| queue_timed_out(state.queue_timeout))?;

    // Caught here, rather than allowed to unwind out of this frame, for two
    // reasons. The extension gets a 500 naming the panic instead of the generic
    // task failure the outer `tokio::spawn` would report; and the permit and the
    // countdown above are then released by an ordinary return, not by unwinding
    // the frame that holds them.
    let (png, snapshot, misses, stages, selected, warnings, rendered_text, refused, edge_hints, pinned) =
        AssertUnwindSafe(run_on_device(
        &state,
        &mut gpu,
        &mut session,
        page,
        &desired,
        plan,
        carried.clone(),
        glossary.clone(),
        script,
        joined_page,
        joined_boundaries,
        translation_seed,
        added_regions,
        removed_regions,
        &placements,
        &pins,
    ))
    .catch_unwind()
    .await
    .unwrap_or_else(|payload| Err(panicked("the translation", &*payload)))?;

    /* Order matters, and this pair used to be 33 lines apart.
     *
     * `TouchOnDrop` was a `_idle` binding, so it ran at the END of this
     * function while the permit was released here -- and everything between is
     * synchronous CPU work (the region walk, the story push). A
     * `free_when_still_idle` that queued on the gate mid-run is woken the
     * instant the permit drops, and it then reads a deadline this page had not
     * yet bumped. That is the race `free_when_still_idle` documents itself as
     * closing, reopened by a binding name.
     *
     * The touch goes FIRST so the deadline is already fresh before any waiter
     * can observe it; releasing first would leave a window, narrow but real,
     * where the unload sees a stale deadline and frees the models this page has
     * just finished paying for.
     *
     * Dropping it here rather than at the return is also the more correct
     * reading of what the countdown measures: it exists to outlast GPU work,
     * and the GPU work ends at the permit, not at the response. Every earlier
     * exit -- the `?` above, a panic -- still drops the guard on its own way
     * out, so nothing below needs to re-arm it. */
    drop(idle_guard);
    drop(gpu);
    if misses.nothing_translated() {
        return Err(nothing_translated(&misses));
    }
    /* Also computed when a story is running, even for a PNG request: the
     * source/translation pairs this page contributes to the next page's prompt
     * come from exactly this walk, and it is a CPU read of a finished scene. */
    /* Unconditional since the drop report, and it used to be
     * `want_regions || story.is_some()`. The condition was an optimisation over
     * a walk that is a few dozen in-memory component reads on a finished
     * snapshot -- unmeasurable beside a page that just spent seconds on the
     * device -- and it bought one blind spot that mattered: `Outcome::dropped`
     * reads this list, so a PNG caller got no `x-birelate-dropped` at all. The
     * drop report is the one channel that can speak on those pages, and having
     * it depend on the response format the caller happened to ask for is
     * exactly the silence it exists to break. */
    let mut regions = regions(&snapshot, page);
    /* Carry the refusal back onto the region it belongs to. The walk above reads
     * the same finished scene the gate patched, so a refused layer is still in
     * this list with its OCR string and its translation -- which is the point:
     * the alternative, deleting it, would leave a caller unable to tell a
     * refusal from a detection that never happened. */
    crate::regions::stamp_refusals(&mut regions, &refused);
    /* Occlusion AFTER the refusal join, because its watermark arm keys on the
     * `refused` field the join just wrote. Observation-only, like everything
     * else in this block: a read with a refused-or-unread box within 16 px is
     * marked suspect on the wire (`occluded_by`), because an occluded glyph
     * once shipped inside a fluent read and nothing said so. */
    crate::regions::stamp_occlusion(
        &mut regions,
        &edge_hints
            .iter()
            .map(|hint| {
                (
                    f64::from(hint.x),
                    f64::from(hint.y),
                    f64::from(hint.width),
                    f64::from(hint.height),
                )
            })
            .collect::<Vec<_>>(),
    );
    /* The drop report is read off `regions` by `Outcome::dropped`, and it has to
     * be read AFTER the refusal join above -- a refused region holds an empty
     * translation too, so a reader that ran before this loop would report every
     * deliberate refusal as a drop. Nothing here consumes it, which is the
     * point: **this patch is observation-only and changes no rendered pixel and
     * no status code.**
     *
     * A page where EVERY region was dropped is a real case and is deliberately
     * left alone: one page, one region of one, is the only instance in
     * 4,768 measured responses. It is not obvious what should
     * happen to it -- refusing hands the reader an error where a readable
     * original page might do, and answering 200 hands them an erased hole and
     * files it in a 24-hour cache -- and a one-in-4,768 case does not get to
     * decide that here, on a patch that ships without a browser session
     * precisely because it cannot change what comes back. It reports `dropped`
     * like any other page; whichever way a later patch decides, it decides with
     * this counter already on the wire to measure against.
     *
     * Note also that all-or-nothing would be the only defensible boundary and
     * even it is not a clean one: the per-page drop fraction across the corpus
     * runs 1.00, 0.87, 0.57, 0.31, 0.25 and on down to 0.04 with no gap
     * anywhere in it, so there is no band to put a constant in. */

    // Only for a JSON caller: a story does not need it (a region with no text
    // contributes no pair) and a PNG caller has nowhere to put it, so the walk
    // is skipped rather than computed and dropped.
    let skipped = if want_regions {
        crate::regions::skipped(&snapshot, page)
    } else {
        Vec::new()
    };
    if let Some(id) = story.as_deref() {
        /* A refused region contributes no pair, and that is not tidiness. The
         * story window is fed verbatim into the next page's prompt, so a
         * hallucinated `ピンク -> PINK` pair would be carried forward as
         * established terminology for up to `--story-pairs` pages -- teaching the
         * model the nonsense it was just spared from drawing. */
        state.stories.record(
            id,
            regions
                .iter()
                /* The refusal half of this filter used to be inline here; it and
                 * the SFX exclusion are ONE composed predicate now,
                 * named so a test can assert on the `&&` rather than on the
                 * halves. `regions::feeds_story` carries both arguments. */
                .filter(|region| crate::regions::feeds_story(region, state.story_excludes_sfx))
                .map(|region| (region.source.clone(), region.translated.clone())),
        );
    }
    let untranslated = crate::regions::untranslated_indices(&regions, &misses.entities);
    // A reported entity with no region behind it should not be possible -- both
    // walks read the same finished scene -- so if the counts part company the
    // JSON is under-reporting and that is worth knowing about.
    if want_regions && untranslated.len() != misses.entities.len() {
        tracing::warn!(
            reported = misses.entities.len(),
            matched = untranslated.len(),
            "some untranslated entities did not match a region"
        );
    }
    /* Done here rather than where the warnings are produced: the renderer sees
     * layers, and the index a caller can act on is a position in *this* region
     * array, which does not exist until the walk above has run. */
    let mut warnings = warnings;
    crate::regions::attribute_warnings(&mut warnings, &regions);
    let mut rendered_text = rendered_text;
    crate::regions::attribute_rendered_text(&mut rendered_text, &regions);
    /* After the attribution above, necessarily: `overflowed` keys on
     * `RenderedText::region`, which does not exist until that join has run. */
    let placement_overflow =
        crate::placement::overflowed(&regions, &rendered_text, &placements);
    Ok(Outcome {
        png,
        regions,
        untranslated,
        skipped,
        misses,
        stages,
        selected,
        warnings,
        rendered_text,
        story_pairs: carried_pairs,
        glossary_terms,
        containment_clause: desired.translation.containment_clause,
        source_script: script,
        edge_hints,
        placement_overflow,
        placement_clamped,
        translation_pins_applied: pinned,
    })
}

/// The 502 for a page where nothing was translated at all.
///
/// The rendered PNG is thrown away deliberately. It is the source text
/// inpainted out and painted back in a translation font -- a page that looks
/// translated and is not, which is the exact failure this whole change exists
/// to stop being silent.
fn nothing_translated(misses: &Misses) -> ApiError {
    ApiError::new(
        StatusCode::BAD_GATEWAY,
        clip(
            // Kept short on purpose: the extension shows this in a 120-byte
            // window, and the cause has to survive inside it.
            &format!(
                "nothing was translated: all {} regions came back in the source language{}",
                misses.segments,
                if misses.truncated {
                    ", cut off at the token cap"
                } else {
                    ""
                }
            ),
            CLIP_BUDGET,
        ),
    )
}

/// The part of a run that can reach koharu's models, and through them libtorch.
///
/// Split out so the frame above never has to unwind. Returns the finished
/// snapshot rather than the regions read off it, so the region walk still happens
/// after the permit is released.
async fn run_on_device(
    state: &AppState,
    gpu: &mut GpuState,
    session: &mut Session,
    page: EntityId,
    desired: &PipelineConfig,
    plan: Plan,
    carried: std::sync::Arc<[koharu_translator::TranslationContext]>,
    glossary: std::sync::Arc<[koharu_translator::TranslationContext]>,
    script: crate::labels::SourceScript,
    // The caller assembled this image so its text ends inside it -- the seam.
    // Lands on the `Request` rather than the config, so a seam costs no reload.
    joined_page: bool,
    joined_boundaries: Vec<f64>,
    // The retry's re-roll, `None` for every caller that did not ask for one.
    translation_seed: Option<u32>,
    // The box editor's lists, empty for every caller but the editor.
    added_regions: Vec<koharu_pipeline::CallerRegion>,
    removed_regions: Vec<koharu_pipeline::CallerRegion>,
    // The editor's placement overrides, borrowed: the caller re-reads them
    // for the overflow report after the render returns.
    placements: &[CallerPlacement],
    // The edit apply's wording pins, applied first in `finish_scene`.
    pins: &[TranslationPin],
) -> Result<
    (
        Vec<u8>,
        Snapshot,
        Misses,
        Vec<StageTiming>,
        Vec<String>,
        Vec<crate::render::LayoutWarning>,
        Vec<crate::render::RenderedText>,
        // Regions refused the page by `labels::hide_implausible`, as
        // (content entity, why). Returned rather than applied here because the
        // region walk that reports them belongs to the caller -- but the gate
        // itself must run on this side, before the render reads the session.
        Vec<(EntityId, crate::labels::Refusal)>,
        // Sub-floor text the detection stage found against a page edge. It
        // reaches the wire and nothing else: no region, no read, no erase.
        Vec<EdgeHintOut>,
        // Regions whose wording an edit apply's pin kept.
        usize,
    ),
    ApiError,
> {
    /* The pre-flight in `translate_page` ran before the gate, and it exempts a
     * warm server -- on the grounds that its weights are already paid for. That
     * ground is false for a page that changes the models: the reload below
     * discards the residency profiles, and the first stage then evicts every
     * model on the card, the local LLM included, so the page runs cold after
     * all. See `vram::after_reload` for the whole mechanism and for why the
     * budget rather than the occupancy is what it compares.
     *
     * Checked here, where `gpu.applied` is the truth and no other request can
     * move it, rather than before the queue where a lock-free copy would be
     * stale by the time the gate was taken. A refusal that lands late is worth
     * more than a native abort with exit code 3 and no message. */
    if needs_reload(&gpu.applied, desired) {
        let headroom = vram::after_reload(&state.vram, state.cold_reserve_bytes);
        if !headroom.sufficient {
            return Err(insufficient_vram_for_reload(&headroom));
        }
    }
    reconcile(&state.pipeline, gpu, desired.clone()).await?;

    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    // The stage list as the pipeline resolves it, not as we spelled it.
    let selected: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    // The translation stage's own report. It is the only place a miss is
    // visible: by the time the scene comes back, an untranslated segment holds
    // its source text under an ordinary `Translation` component and reads
    // exactly like a translated one.
    let misses = Arc::new(std::sync::Mutex::new(Misses::default()));
    let timings: Arc<std::sync::Mutex<Vec<StageTiming>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    /* The detection stage's own report, and the only place it is visible: a
     * sub-floor box is deliberately never written into the scene, so by the time
     * the snapshot comes back there is nothing left of it to walk. */
    let edge_hints: Arc<std::sync::Mutex<Vec<EdgeHintOut>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink: ProgressSink = {
        let observed = observed.clone();
        let misses = misses.clone();
        let timings = timings.clone();
        let selected = selected.clone();
        let edge_hints = edge_hints.clone();
        Arc::new(move |event| match event {
            Progress::Started { stages, .. } => {
                if let Ok(mut selected) = selected.lock() {
                    *selected = stages.iter().map(ToString::to_string).collect();
                }
            }
            Progress::Loading { stage, model, .. } => {
                if let Ok(mut observed) = observed.lock() {
                    observed.push((stage, model));
                }
            }
            Progress::Untranslated {
                entities,
                segments,
                truncated,
                duplicate_ids,
                out_of_range_ids,
                cut_found,
                still_cut,
                ..
            } => {
                if let Ok(mut misses) = misses.lock() {
                    *misses = Misses {
                        entities,
                        segments,
                        truncated,
                        duplicate_ids,
                        out_of_range_ids,
                        cut_found,
                        still_cut,
                    };
                }
            }
            Progress::EdgeHints { hints, .. } => {
                if let Ok(mut edge_hints) = edge_hints.lock() {
                    *edge_hints = hints.into_iter().map(EdgeHintOut::from).collect();
                }
            }
            Progress::Finished {
                stage,
                model,
                elapsed,
                waited,
                ..
            } => {
                if let Ok(mut timings) = timings.lock() {
                    timings.push(StageTiming {
                        stage: stage.to_string(),
                        model,
                        ms: elapsed.as_millis() as u64,
                        wait_ms: waited.as_millis() as u64,
                    });
                }
            }
            _ => {}
        })
    };

    let snapshot = session.snapshot();
    let mut committer = SessionCommitter(session);
    // Awaited directly, never through spawn_blocking: this is an async fn whose
    // internal scheduler has to keep making progress on this runtime, and it
    // already serializes stage work behind its own lock.
    state
        .pipeline
        .execute(
            snapshot,
            Request {
                operation: plan.operation(),
                scope: Scope::Pages(vec![page]),
                progress: Some(sink),
                /* Hung off the *request* rather than TranslationConfig on
                 * purpose: config is what `reconcile` compares, so context that
                 * changes every page would reload the pipeline every page and
                 * throw away the residency profiles `Pipeline::unload_models`
                 * exists to preserve. */
                context: carried.clone(),
                // Same carriage as `context`, fed from the per-series store.
                glossary: glossary.clone(),
                /* Hung off the request for exactly the reason the note above
                 * gives. A seam already costs a third pipeline run per boundary;
                 * making it reload the pipeline as well would be worse than the
                 * refusal it exists to lift. */
                joined_page,
                joined_boundaries,
                /* The retry's one-run seed. On the request like everything
                 * above it; `None` keeps the pipeline's own generation and
                 * with it the fixed constant -- see `Request::translation_seed`. */
                translation_seed,
                // The reader's box edits, one-shot per request.
                // Cloned: `finish_scene` re-reads the added rects to give
                // each caller-touched region a reader-authored frame, after
                // the pipeline has settled them.
                added_regions: added_regions.clone(),
                removed_regions,
                ..Request::default()
            },
            &mut committer,
        )
        .await
        .map_err(|error| stage_failure(&error))?;

    if let Ok(observed) = observed.lock() {
        verify_applied(&observed, desired);
    }

    /* House style and the refusal gates, applied to the finished scene before
     * it is rendered. Extracted VERBATIM into `apply_scene_gates` -- same seven
     * passes, same order, same comments -- so the chain is reachable by a test
     * that builds a `Session` and a `GateSettings` without a `Pipeline`
     * (`AppState` was the blocker, never `Session`). It
     * runs here rather than after the snapshot below because the render reads
     * whatever the session holds at that moment. `finish_scene` wraps the
     * gates and the placement pass so the pair's wiring is itself reachable
     * by a test (test what the caller calls). */
    let (refused, _placed, pinned) = finish_scene(
        &GateSettings::of(state),
        session,
        page,
        script,
        plan,
        placements,
        pins,
        &added_regions,
    );

    // The render tail is genuinely synchronous, so this one does belong on a
    // blocking thread.
    let snapshot = session.snapshot();
    let renderers = state.renderers.clone();
    let font_families = state.font_families.clone();
    let hyphenation = state.hyphenation;
    let size_coherence = state.size_coherence;
    let collision_relief = state.collision_relief;
    let edge_anchored_lettering = state.edge_anchored_lettering;
    let for_render = snapshot.clone();
    // The renderer's own verdict on every layer it set, which render_png used to
    // compute and discard.
    let rendered = tokio::task::spawn_blocking(move || {
        renderers.render_png(
            &for_render,
            page,
            &font_families,
            hyphenation,
            size_coherence,
            collision_relief,
            edge_anchored_lettering,
        )
    })
    .await
    .map_err(|error| task_failed("rendering the page", error))?
    .map_err(|error| {
        // The renderer's own Display is terse and hides the cause in `source`.
        ApiError::internal(clip(
            &format!("render failed: {}", error.root_cause()),
            CLIP_BUDGET,
        ))
    })?;

    let misses = misses.lock().map(|misses| misses.clone()).unwrap_or_default();
    let timings = timings.lock().map(|timings| timings.clone()).unwrap_or_default();
    let selected = selected
        .lock()
        .map(|selected| selected.clone())
        .unwrap_or_default();
    let edge_hints = edge_hints
        .lock()
        .map(|hints| hints.clone())
        .unwrap_or_default();
    Ok((
        rendered.png,
        snapshot,
        misses,
        timings,
        selected,
        rendered.warnings,
        rendered.text,
        refused,
        edge_hints,
        pinned,
    ))
}

/// The scene-finishing pair between `Pipeline::execute` and the render: the
/// seven gates, then the reader's placement pass -- placement last, because
/// there is no point placing a layer a gate just hid.
///
/// Extracted for `apply_scene_gates`' own reason, one level up: the WIRING of
/// the pair can ship unwired while each half's own tests stay green, and a
/// function a test can call with a bare `Session` + `GateSettings` is what
/// makes that impossible to repeat silently.
fn finish_scene(
    settings: &GateSettings<'_>,
    session: &mut Session,
    page: EntityId,
    script: crate::labels::SourceScript,
    plan: Plan,
    placements: &[CallerPlacement],
    pins: &[TranslationPin],
    added_regions: &[koharu_pipeline::CallerRegion],
) -> (Vec<(EntityId, crate::labels::Refusal)>, usize, usize) {
    /* Pins FIRST, so every gate judges the wording that will
     * actually ship: the duplicate gate must see the pinned strings it saw
     * on the original run, and uppercasing is idempotent on a stored value
     * that already went through it. The caller frames come after
     * the gates -- no point framing a layer a gate just hid -- and BEFORE
     * the placements, which overwrite the same component and must win. */
    let pinned = crate::pins::apply_translation_pins(session, page, pins);
    let refused = apply_scene_gates(settings, session, page, script, plan);
    crate::placement::assert_caller_frames(session, page, added_regions);
    let placed = crate::placement::apply_caller_placement(session, page, placements);
    (refused, placed, pinned)
}

/// The scalars the seven scene gates read, split off `Shared` so the chain is
/// constructible in a test without a `Pipeline`.
///
/// `run_on_device`'s gate chain was
/// unreachable by any test because every gate read `&AppState`, and `Shared`
/// owns a `Pipeline` no unit test can build -- while `Session` was
/// constructible in-crate all along (`labels.rs`'s fixtures prove it). This
/// struct is the cut that makes the chain testable; it carries no behaviour.
struct GateSettings<'a> {
    uppercase_dialogue: bool,
    skip_implausible_text: bool,
    scope_watermark_refusals: bool,
    leave_misread_bubbles: bool,
    korean_script_strict: bool,
    skip_duplicate_text: bool,
    duplicate_oriented_overlap: bool,
    duplicate_shared_source: bool,
    sfx_dictionary: &'a crate::sfx::Dictionary,
    fit_free_text: bool,
    source_ink_fraction: f32,
}

impl<'a> GateSettings<'a> {
    fn of(state: &'a AppState) -> Self {
        Self {
            uppercase_dialogue: state.uppercase_dialogue,
            skip_implausible_text: state.skip_implausible_text,
            scope_watermark_refusals: state.scope_watermark_refusals,
            leave_misread_bubbles: state.leave_misread_bubbles,
            korean_script_strict: state.korean_script_strict,
            skip_duplicate_text: state.skip_duplicate_text,
            duplicate_oriented_overlap: state.duplicate_oriented_overlap,
            duplicate_shared_source: state.duplicate_shared_source,
            sfx_dictionary: &state.sfx_dictionary,
            fit_free_text: state.fit_free_text,
            source_ink_fraction: state.source_ink_fraction,
        }
    }
}

/// The seven scene passes `run_on_device` applies between `Pipeline::execute`
/// and the render, moved here VERBATIM -- same passes, same order, same
/// comments. The order is load-bearing and characterization-tested below:
/// the merged `refused` list is stamped onto the regions in order
/// (`regions::stamp_refusals` is last-write-wins), so a reordering silently
/// rewrites the refusal reason on the wire while changing no pixel.
fn apply_scene_gates(
    settings: &GateSettings<'_>,
    session: &mut Session,
    page: EntityId,
    script: crate::labels::SourceScript,
    plan: Plan,
) -> Vec<(EntityId, crate::labels::Refusal)> {
    /* House style. Cheap, CPU-only and best-effort: a page whose scene will
     * not take the patch still renders, just in mixed case. */
    if settings.uppercase_dialogue && plan.letters_text() {
        crate::lettering::uppercase_dialogue(session, page);
    }

    /* Refuse the page to a region whose OCR text is not text. Same seam and the
     * same best-effort contract as the pass above, and it must run BEFORE the
     * caller's render snapshot, because hiding a layer after the render has
     * read the session would report a refusal that did not happen. */
    let refused = if settings.skip_implausible_text && plan.letters_text() {
        let refused = crate::labels::hide_implausible(
            session,
            page,
            script,
            settings.scope_watermark_refusals,
            settings.leave_misread_bubbles,
            settings.korean_script_strict,
        );
        if !refused.is_empty() {
            /* Logged, because the area-vs-page erase gate is silent: its mask
             * arm refuses to erase and says nothing at all, so the only evidence
             * it ever fired is a pixel diff -- which is how the defect this gate fixes went
             * unnoticed through a five-arm sweep. Name the count, the script and
             * the page, so a refusal can be told from a misapplication. */
            let mut counts: std::collections::BTreeMap<&'static str, usize> =
                std::collections::BTreeMap::new();
            for (_content, refusal) in &refused {
                *counts.entry(refusal.why()).or_default() += 1;
            }
            tracing::info!(
                regions = refused.len(),
                source_script = ?script,
                reasons = ?counts,
                page = %page,
                "refusing to letter regions whose text is not text"
            );
        }
        refused
    } else {
        Vec::new()
    };

    /* One utterance the detector found twice, lettered twice, on top of itself.
     *
     * Runs AFTER `hide_implausible` and merges into its list, so the two gates
     * report through the one `refused` field. The order is load-bearing and the
     * walk skips layers that are already hidden: without that it re-refuses a
     * region the first gate had already refused, and since the merged list is
     * stamped onto the regions in order, the duplicate reason would overwrite the
     * true one in `format=json`. Measured, and fixed in `duplicate.rs`. */
    let mut refused = refused;

    /* A region no OCR engine was ever asked to read.
     *
     * UNGATED, unlike every other arm here, and that is the point. The two
     * gates above are lettering decisions and both hang off `plan.letters_text()`;
     * this one reports something the *reader* stage did, under
     * `--skip-implausible-regions`, before lettering was ever considered. Gating
     * it on the lettering flags would make the report vanish exactly when
     * somebody turned lettering off to find out what happened.
     *
     * Merged last but provably collision-free: `unread_regions` fires only when
     * there is no `SourceText`, and every other refusal in this list requires one
     * -- `classify` takes the string, and a duplicate needs a translation. The
     * `already` filter is belt-and-braces for that invariant rather than a fix for
     * an observed clash, and it is here because the merged list is stamped onto
     * the regions in order, so a wrong entry would silently overwrite a true one.
     * That failure is not hypothetical: the block above documents it happening
     * between the first two gates. */
    {
        let already: std::collections::HashSet<_> =
            refused.iter().map(|(content, _)| *content).collect();
        let unread = crate::labels::unread_regions(&session.snapshot(), page);
        let unread: Vec<_> = unread
            .into_iter()
            .filter(|(content, _)| !already.contains(content))
            .collect();
        if !unread.is_empty() {
            tracing::info!(
                regions = unread.len(),
                page = %page,
                "naming regions no OCR engine was asked to read"
            );
            refused.extend(unread);
        }
    }

    if settings.skip_duplicate_text && plan.letters_text() {
        let duplicates = crate::duplicate::hide_duplicate_lettering(
            session,
            page,
            settings.duplicate_oriented_overlap,
            settings.duplicate_shared_source,
        );
        if !duplicates.is_empty() {
            tracing::info!(
                regions = duplicates.len(),
                page = %page,
                "refusing the redundant half of duplicate-lettered regions"
            );
            refused.extend(
                duplicates
                    .into_iter()
                    .map(|content| (content, crate::labels::Refusal::DuplicateLettering)),
            );
        }
    }
    let refused = refused;

    /* Pinned English for drawn sound effects, same place and same rationale.
     * Gated on the detector's onomatopoeia label inside `pin_sound_effects`, so
     * a two-kana dictionary entry cannot rewrite a spoken line. */
    if plan.letters_text() && !settings.sfx_dictionary.is_empty() {
        let pinned = crate::sfx::pin_sound_effects(session, page, settings.sfx_dictionary);
        if pinned > 0 {
            tracing::debug!(pinned, "sound effects pinned from the dictionary");
        }
    }

    /* Only when the artwork was kept. The detection stage withholds the halo from
     * in-bubble text because it normally sits on a flat inpainted fill -- true on
     * every ordinary page, and false on exactly this one, where the balloon was
     * never erased and the English lands on the Japanese. Runs here, on the same
     * session and for the same reason as the uppercasing above: how the result is
     * *set* is a house style, and keeping it here costs no change to koharu. */
    if plan.kept_art() {
        crate::lettering::halo_bubble_text(session, page);
    }

    /* Stops a caption sitting on artwork from being set four times as heavy as
     * the Japanese it replaces. Auto-fit fills whatever box it is given, and
     * the detection stage deliberately widens a vertical caption's box to escape an
     * illegible 6-10px setting -- so on those layers the two compound. Measured
     * on the title page before this: 55 characters at 73.4px, 4.2x the source's
     * ink. Same placement as the passes above, and for the same reason. */
    if settings.fit_free_text && plan.letters_text() {
        crate::lettering::fit_free_text(session, page, settings.source_ink_fraction);
    }

    refused
}

/// Which stages a request wants, as one exhaustive choice.
///
/// Two independent booleans would admit a fourth state that means nothing --
/// "skip the eraser AND run only the eraser" -- and would leave every call site
/// re-deriving the same precedence. One enum makes the impossible combination
/// unrepresentable past the point it is parsed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Plan {
    /// Detection, OCR, translation, inpainting. What a reader gets.
    Full,
    /// Everything but inpainting: English over untouched artwork.
    KeepArt,
    /// Detection and inpainting only: the cleaned page, no text at all.
    CleanOnly,
    /// Detection alone: the boxes and the sub-floor hints, and nothing else.
    ///
    /// For the seam lookahead. `extension/seam.js`'s `seamEdges` is
    /// computed from GEOMETRY ALONE — it reads `seamEffectiveBox`, x/y/width/
    /// height, and never touches `source` or `translated` — and both `regions`
    /// and `edge_hints` come out of detection. So deciding every boundary in a
    /// strip before the reader reaches it needs this stage and no other.
    ///
    /// The cheap-looking alternatives are not cheap: `CleanOnly` adds inpainting
    /// (~617 ms) and returns a cleaned page, `KeepArt` adds OCR (~442 ms) and
    /// translation (~1,275 ms), against detection's ~119 ms.
    DetectOnly,
}

impl Plan {
    fn operation(self) -> Operation {
        match self {
            Self::Full => Operation::Full,
            Self::KeepArt => keep_the_art(),
            /* `Through { Inpainting }` resolves to [Detection, Inpainting] --
             * note it does NOT include OCR, which is exactly what makes the page
             * come back blank. With no `SourceText` in the scene the renderer's
             * `fallback_to_source_text` has nothing to fall back to; ask for OCR
             * as well and it re-typesets the Japanese over itself in Arial. */
            Self::CleanOnly => Operation::Through {
                stage: Stage::Inpainting,
            },
            /* Spelled as an explicit list rather than `Through { Detection }` for
             * the same reason `keep_the_art` is: the list says what it means and
             * cannot quietly change meaning upstream. It happens to expand to the
             * same single stage today. */
            Self::DetectOnly => Operation::Stages {
                stages: vec![Stage::Detection],
            },
        }
    }

    /// Whether the balloons kept their original pixels, and so whether in-bubble
    /// text needs the halo the detection stage withholds from it.
    ///
    /// True for `DetectOnly` as well, which runs no inpainting stage. It is inert
    /// there — that plan letters nothing, so the halo question never arises — but
    /// answering `false` would be a lie waiting for the first caller that letters
    /// anything.
    fn kept_art(self) -> bool {
        matches!(self, Self::KeepArt | Self::DetectOnly)
    }

    /// Whether any text is rendered at all. `CleanOnly` and `DetectOnly` have none
    /// to letter — neither runs OCR, so neither has a `SourceText` the renderer's
    /// `fallback_to_source_text` could re-typeset.
    fn letters_text(self) -> bool {
        !matches!(self, Self::CleanOnly | Self::DetectOnly)
    }
}

/// The plan a request asked for, or a 400 naming the contradiction.
///
/// **Extracted from the handler so the exclusion is testable at all.** It used to
/// sit inline, where only a full request could reach it — so the one guard that
/// turns a contradictory request into a 400 had no test, on a rule whose whole job
/// is to refuse. Three flags admit four meaningless combinations and one enum
/// makes them unrepresentable past this point.
fn choose_plan(
    skip_inpainting: bool,
    clean_only: bool,
    detect_only: bool,
) -> Result<Plan, ApiError> {
    /* Counted rather than enumerated as pairs: with three flags there are four
     * bad combinations, and a chain of `&&` checks is where the fourth gets
     * forgotten. */
    let asked = usize::from(skip_inpainting) + usize::from(clean_only) + usize::from(detect_only);
    if asked > 1 {
        return Err(ApiError::bad_request(
            "skip_inpainting, clean_only and detect_only ask for different pipelines; \
             send at most one",
        ));
    }
    Ok(if clean_only {
        Plan::CleanOnly
    } else if detect_only {
        Plan::DetectOnly
    } else if skip_inpainting {
        Plan::KeepArt
    } else {
        Plan::Full
    })
}

/// The retry button's re-roll knob, or a 400 naming the garbage.
///
/// Extracted from the handler for `choose_plan`'s reason: the refusal is the
/// point, and a refusal that only a full request can reach has no test. The
/// failure this guards against is silent, not loud -- a seed that fell back
/// to "absent" would keep the fixed sampler constant and answer the retry
/// with the byte-identical page it was asked to replace, and
/// nothing on the wire would say so.
fn parse_seed(value: Option<&str>) -> Result<Option<u32>, ApiError> {
    match value.map(str::trim) {
        None => Ok(None),
        Some(text) => text.parse::<u32>().map(Some).map_err(|_| {
            ApiError::bad_request("seed must be a decimal unsigned 32-bit integer")
        }),
    }
}

/// The largest box list one request may carry. A page has tens of regions at
/// most; a list past this is a caller defect, not a bigger page.
const CALLER_REGION_CAP: usize = 32;

/// The box editor's rectangles, parsed at the edge like every declared field.
/// JSON with named fields rather than a comma list, so a box is
/// unreadable exactly when it is wrong; `deny_unknown_fields` so a misspelled
/// `widht` is a 400 today instead of a box of height 0 forever. Refused
/// rather than skimmed for `joined_boundaries`' reason, sharpened: a
/// half-parsed removal leaves a deleted box erasing its artwork, and a
/// half-parsed addition silently drops the reader's own correction.
fn parse_caller_regions(
    field: &str,
    value: Option<&str>,
) -> Result<Vec<koharu_pipeline::CallerRegion>, ApiError> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct WireBox {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    }
    let Some(text) = value.map(str::trim).filter(|text| !text.is_empty()) else {
        return Ok(Vec::new());
    };
    let boxes: Vec<WireBox> = serde_json::from_str(text).map_err(|_| {
        ApiError::bad_request(format!(
            "{field} must be a JSON array of {{x, y, width, height}} boxes"
        ))
    })?;
    if boxes.len() > CALLER_REGION_CAP {
        return Err(ApiError::bad_request(format!(
            "{field} carries {} boxes; the cap is {CALLER_REGION_CAP}",
            boxes.len()
        )));
    }
    for wire in &boxes {
        let finite = [wire.x, wire.y, wire.width, wire.height]
            .iter()
            .all(|v| v.is_finite());
        if !finite || wire.width <= 0.0 || wire.height <= 0.0 {
            return Err(ApiError::bad_request(format!(
                "{field} boxes need finite coordinates and positive width and height"
            )));
        }
    }
    Ok(boxes
        .into_iter()
        .map(|wire| koharu_pipeline::CallerRegion {
            x: wire.x,
            y: wire.y,
            width: wire.width,
            height: wire.height,
        })
        .collect())
}

/// The smallest placement box a reader may draw, in source pixels. Below ~8
/// the renderer's `text_inset` ([4.0; 4]) leaves a zero-area layout box and
/// `render_layer` refuses -- a 500 for the whole page. 24 clears the inset
/// plus a 9px floor line at 1.2 leading, with slack. Mirrored by the editor's
/// `BOXEDIT_MIN_PLACEMENT` so the reader hears it there first.
const PLACEMENT_MIN_SPAN: f32 = 24.0;

/// A caller's hard boundary for one region's English: `target`
/// is the region rect the editor displayed, matched centre-inside against
/// the settled regions; `place` is where that region's English must fit.
#[derive(Debug)]
pub struct CallerPlacement {
    pub target: koharu_pipeline::CallerRegion,
    pub place: koharu_pipeline::CallerRegion,
}

/// The placement list, parsed at the edge exactly like `parse_caller_regions`
/// and refused-never-skimmed for a sharper reason: a half-parsed placement
/// letters English somewhere the reader did not choose, and nothing on the
/// wire says so.
fn parse_caller_placements(value: Option<&str>) -> Result<Vec<CallerPlacement>, ApiError> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct WirePlacement {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        place_x: f32,
        place_y: f32,
        place_width: f32,
        place_height: f32,
    }
    let Some(text) = value.map(str::trim).filter(|text| !text.is_empty()) else {
        return Ok(Vec::new());
    };
    let entries: Vec<WirePlacement> = serde_json::from_str(text).map_err(|_| {
        ApiError::bad_request(
            "regions_place must be a JSON array of \
             {x, y, width, height, place_x, place_y, place_width, place_height} entries",
        )
    })?;
    if entries.len() > CALLER_REGION_CAP {
        return Err(ApiError::bad_request(format!(
            "regions_place carries {} entries; the cap is {CALLER_REGION_CAP}",
            entries.len()
        )));
    }
    for wire in &entries {
        let finite = [
            wire.x,
            wire.y,
            wire.width,
            wire.height,
            wire.place_x,
            wire.place_y,
            wire.place_width,
            wire.place_height,
        ]
        .iter()
        .all(|v| v.is_finite());
        if !finite || wire.width <= 0.0 || wire.height <= 0.0 {
            return Err(ApiError::bad_request(
                "regions_place entries need finite coordinates and positive spans",
            ));
        }
        if wire.place_width < PLACEMENT_MIN_SPAN || wire.place_height < PLACEMENT_MIN_SPAN {
            return Err(ApiError::bad_request(format!(
                "a placement box must be at least {PLACEMENT_MIN_SPAN} source pixels each way"
            )));
        }
    }
    Ok(entries
        .into_iter()
        .map(|wire| CallerPlacement {
            target: koharu_pipeline::CallerRegion {
                x: wire.x,
                y: wire.y,
                width: wire.width,
                height: wire.height,
            },
            place: koharu_pipeline::CallerRegion {
                x: wire.place_x,
                y: wire.place_y,
                width: wire.place_width,
                height: wire.place_height,
            },
        })
        .collect())
}

/// The largest pin list one request may carry: a page has tens of regions.
const TRANSLATION_PIN_CAP: usize = 128;
/// Per-string ceilings, sized generously over the longest measured region
/// strings; past them the list is a caller defect, not a longer bubble.
const TRANSLATION_PIN_SOURCE_MAX: usize = 2048;
const TRANSLATION_PIN_TEXT_MAX: usize = 4096;

/// An edit apply's wording pin: a region whose OCR source equals
/// `source` byte-identically keeps `translation` instead of the model's
/// fresh draw.
#[derive(Debug)]
pub struct TranslationPin {
    pub source: String,
    pub translation: String,
}

/// The pin list, refused-never-skimmed with this family's sharpest stake yet:
/// a skimmed pin silently re-rolls a bubble the reader believed they were
/// keeping, and the reworded page carries nothing that says so.
fn parse_translation_pins(value: Option<&str>) -> Result<Vec<TranslationPin>, ApiError> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct WirePin {
        s: String,
        t: String,
    }
    let Some(text) = value.map(str::trim).filter(|text| !text.is_empty()) else {
        return Ok(Vec::new());
    };
    let entries: Vec<WirePin> = serde_json::from_str(text).map_err(|_| {
        ApiError::bad_request("translation_pins must be a JSON array of {s, t} pairs")
    })?;
    if entries.len() > TRANSLATION_PIN_CAP {
        return Err(ApiError::bad_request(format!(
            "translation_pins carries {} pairs; the cap is {TRANSLATION_PIN_CAP}",
            entries.len()
        )));
    }
    for wire in &entries {
        if wire.s.trim().is_empty() || wire.t.trim().is_empty() {
            return Err(ApiError::bad_request(
                "translation_pins pairs need non-blank source and translation",
            ));
        }
        if wire.s.len() > TRANSLATION_PIN_SOURCE_MAX || wire.t.len() > TRANSLATION_PIN_TEXT_MAX {
            return Err(ApiError::bad_request(
                "a translation_pins pair is longer than any real region string",
            ));
        }
    }
    Ok(entries
        .into_iter()
        .map(|wire| TranslationPin {
            source: wire.s,
            translation: wire.t,
        })
        .collect())
}

/// Every stage but inpainting, so the English is drawn over untouched artwork.
///
/// Nothing else in the pipeline has to change for this to work. Inpainting
/// writes its result as a separate `Cleanup` child layer and never overwrites
/// the page's own `source` asset, so with the stage absent the compositor has no
/// cleanup layer to draw and the base image is the original bytes. Render is not
/// a stage at all -- the server runs it after `execute` returns -- so the text
/// still lands, over artwork nothing touched.
///
/// Spelled as an explicit list rather than `Through { stage: Translation }`,
/// which expands to the same three stages today. The list says what it means and
/// cannot quietly change meaning upstream.
fn keep_the_art() -> Operation {
    Operation::Stages {
        stages: vec![Stage::Detection, Stage::Ocr, Stage::Translation],
    }
}

/// The queue is what times out, never a running job: once a translation starts
/// it has to finish, or the models it loaded were paid for and thrown away.
fn queue_timed_out(waited: Duration) -> ApiError {
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        format!(
            "server busy: the GPU queue did not free up in {}s; retry when the current page \
             finishes",
            waited.as_secs()
        ),
    )
}

/// Takes `defaults` and `pinned` rather than `&AppState` so a test can call
/// the very function `translate` does: `Shared` owns a `Pipeline` no unit test
/// can build.
fn selection_from(
    form: &TranslateForm,
    defaults: &Defaults,
    pinned: &Pinned,
) -> Result<Selection, ApiError> {
    let ocr = match form.ocr.as_deref() {
        Some(value) => models::resolve_ocr(value, defaults.ocr_substitute.as_ref())?,
        None => defaults.ocr.clone(),
    };
    /* THE ONE RESOLUTION SITE. It moved here from `translate` when the script
     * began reaching the pipeline as well as the labeller: the erase veto needs
     * it inside `PipelineConfig`, which is built from `Selection`, so resolving
     * it later would have meant resolving it twice. `translate` now reads it back
     * off the selection. The warning that made this worth care is still the one
     * that was written there: a second resolution site is a second place for the
     * two to disagree about what language the page is in. */
    let source_script = crate::labels::SourceScript::resolve(
        form.source_language.as_deref(),
        Some(crate::models::ocr_model_name(&ocr)),
    );
    Ok(Selection {
        ocr,
        source_script,
        inpainting: match form.inpainting.as_deref() {
            Some(value) => models::parse_inpainting(value, defaults.rorem_steps, defaults.flux_strength)?,
            None => defaults.inpainting.clone(),
        },
        target_language: match form.target_language.as_deref() {
            Some(value) => models::parse_language(value)?,
            None => defaults.target_language,
        },
        // Settled before any work is done: asking the local engine for a
        // different model at runtime is what aborts the process on a full card.
        llm: pinned.resolve(form.provider.as_deref(), form.llm.as_deref())?,
        /* Absent falls back to the process default, so every caller that
         * predates the popup toggle keeps the behaviour it had. Parsed
         * strictly like every other flag here: an unexpected spelling is a
         * 400, never a silent off, because reading "yes" as false would turn
         * a setting the reader deliberately enabled into a no-op they cannot
         * see. */
        segment_context: match form.segment_context.as_deref() {
            Some(value) => models::parse_flag("segment_context", value)?,
            None => defaults.segment_context,
        },
    })
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt as _;

    use super::*;
    use crate::{error::BUDGET, guard::digest};

    const HOST: &str = "127.0.0.1:8765";

    // ------------------------------------------------------------------- //
    // `apply_scene_gates` -- characterization tests.
    //
    // These pin the ORDER of the seven passes, which is invisible to pixels:
    // the merged `refused` list is stamped last-write-wins, so a reordering
    // rewrites the wire's refusal reason while rendering the identical page.
    // Each test was proven able to go red by deliberately breaking the thing
    // it watches (the break and the red are named per test), per the
    // prove-a-check-can-fail rule.
    // ------------------------------------------------------------------- //

    use koharu_scene::{
        At, Authored, Geometry, Origin, PageDraft, RecognizedFrom, SourceText, TextLayout,
        TextLayoutKind, TextRegion, TextRole, Translation, Typography,
    };

    /// Everything ON that the chain reads, dictionary supplied by the caller.
    /// `uppercase_dialogue` is deliberately ON (it ships OFF) because two of
    /// these tests pin its position in the order; `fit_free_text` is OFF so
    /// the fixture needs no fittable free-text geometry.
    fn gate_settings(dictionary: &crate::sfx::Dictionary) -> GateSettings<'_> {
        GateSettings {
            uppercase_dialogue: true,
            skip_implausible_text: true,
            scope_watermark_refusals: false,
            leave_misread_bubbles: false,
            korean_script_strict: false,
            skip_duplicate_text: true,
            duplicate_oriented_overlap: true,
            duplicate_shared_source: false,
            sfx_dictionary: dictionary,
            fit_free_text: false,
            source_ink_fraction: 0.12,
        }
    }

    /// One page, N text layers. Each spec is (source-or-None, translation,
    /// role, region label, painted x, typography size) -- geometry is set on
    /// the LAYER too, because the duplicate gate compares where the English
    /// was painted, not where the source was read. A `Some` size writes a
    /// `Typography` onto the layer (`auto_fit: true`, so the component
    /// validates with or without it), which is what the halo gate requires;
    /// `None` keeps the layer bare, as every fixture before the KeepArt
    /// tests was.
    fn page_with_layers(
        specs: &[(Option<&str>, Option<&str>, &str, &str, f64, Option<f32>)],
    ) -> (Session, EntityId, Vec<EntityId>, Vec<EntityId>) {
        let mut session = Session::memory().expect("an in-memory session");
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 844.0, 1200.0), At::End)?;
                let mut contents = Vec::new();
                let mut layers = Vec::new();
                for (source, translation, role, label, x, typography_size) in specs {
                    let region = edit.add_analysis_region::<TextRegion>(
                        page,
                        At::End,
                        &Geometry::rectangle(*x, 100.0, 200.0, 150.0),
                        Some((*label).to_owned()),
                    )?;
                    let content = edit.add_text_content(page, At::End)?;
                    if let Some(source) = source {
                        edit.set(
                            content,
                            &SourceText {
                                text: Authored::user((*source).to_owned()),
                                language: None,
                            },
                        )?;
                    }
                    if let Some(translation) = translation {
                        edit.set(
                            content,
                            &Translation {
                                text: Authored::user((*translation).to_owned()),
                                language: None,
                            },
                        )?;
                    }
                    edit.set(
                        content,
                        &TextRole {
                            origin: Origin::User,
                            role: (*role).to_owned(),
                        },
                    )?;
                    let layer = edit.add_text_layer(
                        page,
                        At::End,
                        content,
                        &TextLayout {
                            origin: Origin::User,
                            kind: TextLayoutKind::Paragraph,
                        },
                    )?;
                    edit.set(layer, &Geometry::rectangle(*x, 100.0, 200.0, 150.0))?;
                    if let Some(size) = typography_size {
                        edit.set(
                            layer,
                            &Typography {
                                origin: Origin::User,
                                preferred_font: None,
                                font_weight: None,
                                size: Some(*size),
                                auto_fit: true,
                                color: None,
                                stroke_color: None,
                                stroke_width: None,
                                alignment: None,
                                writing_mode: None,
                                extensions: Default::default(),
                            },
                        )?;
                    }
                    edit.relate::<RecognizedFrom>(content, region)?;
                    contents.push(content);
                    layers.push(layer);
                }
                ids = Some((page, contents, layers));
                Ok(())
            })
            .expect("the fixture scene is valid");
        session.commit(patch).expect("the fixture scene commits");
        let (page, contents, layers) = ids.expect("the edit ran to completion");
        (session, page, contents, layers)
    }

    fn refused_on_the_wire(
        session: &Session,
        page: EntityId,
        refused: &[(EntityId, crate::labels::Refusal)],
        content: EntityId,
    ) -> Option<&'static str> {
        let snapshot = session.snapshot();
        let mut regions = crate::regions::regions(&snapshot, page);
        crate::regions::stamp_refusals(&mut regions, refused);
        regions
            .iter()
            .find(|region| region.content == content)
            .unwrap_or_else(|| panic!("{content} is missing from regions[]"))
            .refused
    }

    /// **The measured incident, as a fixture** -- from a manhua test page: a
    /// punctuation-only region overlapping a real utterance with the
    /// IDENTICAL translation. `hide_implausible` must refuse it FIRST, and the
    /// duplicate gate must then skip the hidden layer, so the wire carries the
    /// true reason and the real utterance survives unrefused.
    ///
    /// Proven able to fail: commenting the `Visibility` skip in
    /// `duplicate.rs`'s candidate walk turns this red with the punctuation
    /// region reported as "duplicate lettering" -- the exact defect the order
    /// exists to prevent.
    #[test]
    fn a_duplicate_reason_never_overwrites_an_earlier_refusal() {
        let dictionary = crate::sfx::Dictionary::default();
        let (mut session, page, contents, _layers) = page_with_layers(&[
            (Some("000"), Some("That's it!"), "dev.koharu.text.free-text", "text", 100.0, None),
            (Some("就是这样！"), Some("That's it!"), "dev.koharu.text.free-text", "text", 120.0, None),
        ]);
        let refused = apply_scene_gates(
            &gate_settings(&dictionary),
            &mut session,
            page,
            crate::labels::SourceScript::Chinese,
            Plan::Full,
        );

        assert_eq!(
            refused.len(),
            1,
            "exactly the punctuation-only side is refused: {refused:?}"
        );
        assert_eq!(refused[0].0, contents[0]);
        assert_eq!(
            refused_on_the_wire(&session, page, &refused, contents[0]),
            Some("no letters, only punctuation or symbols"),
            "the TRUE reason reaches the wire, not the duplicate gate's"
        );
        assert_eq!(
            refused_on_the_wire(&session, page, &refused, contents[1]),
            None,
            "the real utterance is not the redundant half of anything"
        );
    }

    /// The dictionary pin lands AFTER the uppercase pass, so pinned lettering
    /// arrives exactly as the dictionary spells it. Swapping passes 1 and 5
    /// turns this red ("Rumble" -> "RUMBLE"): the pin would run first and be
    /// a no-op (the translation already matches), and the uppercase pass
    /// would then shout over it.
    #[test]
    fn a_pinned_sound_effect_is_not_uppercased() {
        let mut dictionary = crate::sfx::Dictionary::default();
        dictionary
            .merge_json(r#"{"entries":[{"source":"轰","target":"Rumble"}]}"#)
            .expect("the pin parses");
        let (mut session, page, contents, _layers) = page_with_layers(&[(
            Some("轰"),
            Some("Rumble"),
            // Dialogue ROLE with an onomatopoeia LABEL: reachable by both
            // passes, which is what makes their order observable at all.
            "dev.koharu.text.dialogue",
            "onomatopoeia",
            100.0,
            None,
        )]);
        apply_scene_gates(
            &gate_settings(&dictionary),
            &mut session,
            page,
            crate::labels::SourceScript::Chinese,
            Plan::Full,
        );

        let translation = session
            .snapshot()
            .component::<Translation>(contents[0])
            .expect("the content is in the scene")
            .expect("the translation survives the gates");
        assert_eq!(
            translation.text.value, "Rumble",
            "the pin is the last writer; uppercase must precede it"
        );
    }

    /// Gate 3 (`unread_regions`, ungated) merges into the refusal list without
    /// disturbing gate 2's entries: one junk region and one never-read region
    /// come back with their own reasons side by side. Proven able to fail by
    /// dropping the `refused.extend(unread)` merge, which loses the unread
    /// entry while every other assertion stays green.
    #[test]
    fn an_unread_region_and_a_refused_region_keep_their_own_reasons() {
        let dictionary = crate::sfx::Dictionary::default();
        let (mut session, page, contents, _layers) = page_with_layers(&[
            (Some("000"), Some("!!"), "dev.koharu.text.free-text", "text", 100.0, None),
            (None, None, "dev.koharu.text.free-text", "text", 500.0, None),
        ]);
        let refused = apply_scene_gates(
            &gate_settings(&dictionary),
            &mut session,
            page,
            crate::labels::SourceScript::Chinese,
            Plan::Full,
        );

        assert_eq!(refused.len(), 2, "one entry per cause: {refused:?}");
        assert_eq!(
            refused_on_the_wire(&session, page, &refused, contents[0]),
            Some("no letters, only punctuation or symbols"),
        );
        assert_eq!(
            refused_on_the_wire(&session, page, &refused, contents[1]),
            Some("no OCR engine ever read it"),
        );
    }

    /// Gate 6 (`halo_bubble_text`) is the one pass keyed on the PLAN rather
    /// than on any `GateSettings` scalar, and no fixture anywhere had ever
    /// exercised it: the three tests above all run `Plan::Full`, and
    /// `lettering.rs`'s own tests only do arithmetic on the constants. Under
    /// `Plan::KeepArt` the dictionary pin still lands -- gate 5 is plan-independent, NOT
    /// order-pinned here: gates 5 and 6 write different components on
    /// different entities and cannot observe each other -- and the dialogue
    /// layer gains the in-bubble halo. The stroke width is asserted as a
    /// recomputation: `28.0 * IN_BUBBLE_HALO_RATIO` is an f32 product
    /// (1.4000001...), and transcribing the decimal `1.4` fails.
    ///
    /// Proven able to fail as a PAIR with the `Plan::Full` control below:
    /// flipping `if plan.kept_art()` to `if !plan.kept_art()` at the gate
    /// turned BOTH red -- this one losing its halo, the control gaining one
    /// -- which is what proves the composed plan predicate, not either half.
    #[test]
    fn keep_art_halos_the_pinned_sound_effect() {
        let mut dictionary = crate::sfx::Dictionary::default();
        dictionary
            .merge_json(r#"{"entries":[{"source":"轰","target":"Rumble"}]}"#)
            .expect("the pin parses");
        let (mut session, page, contents, layers) = page_with_layers(&[(
            Some("轰"),
            Some("Rumble"),
            "dev.koharu.text.dialogue",
            "onomatopoeia",
            100.0,
            Some(28.0),
        )]);
        apply_scene_gates(
            &gate_settings(&dictionary),
            &mut session,
            page,
            crate::labels::SourceScript::Chinese,
            Plan::KeepArt,
        );

        let translation = session
            .snapshot()
            .component::<Translation>(contents[0])
            .expect("the content is in the scene")
            .expect("the translation survives the gates");
        assert_eq!(
            translation.text.value, "Rumble",
            "gate 5 is plan-independent: the pin lands under KeepArt too"
        );

        let typography = session
            .snapshot()
            .component::<Typography>(layers[0])
            .expect("the layer is in the scene")
            .expect("the fixture set a typography");
        assert_eq!(
            typography.stroke_color,
            Some(crate::lettering::IN_BUBBLE_HALO_COLOR),
            "kept art has no flat fill under the glyphs, so the halo must appear"
        );
        assert_eq!(
            typography.stroke_width,
            Some(28.0f32 * crate::lettering::IN_BUBBLE_HALO_RATIO),
        );
    }

    /// The negative control for the halo: the identical fixture under
    /// `Plan::Full` keeps its typography untouched, because inpainting lays a
    /// flat fill under the glyphs and the detection stage already refuses in-bubble
    /// halos there. Without this control the positive test above stays green
    /// with the plan gate deleted outright -- an unconditional halo passes it.
    /// Its red is documented on the test above: the same `!plan.kept_art()`
    /// flip turns this one red by GAINING a halo.
    #[test]
    fn a_full_plan_leaves_the_bubble_text_unhaloed() {
        let mut dictionary = crate::sfx::Dictionary::default();
        dictionary
            .merge_json(r#"{"entries":[{"source":"轰","target":"Rumble"}]}"#)
            .expect("the pin parses");
        let (mut session, page, _contents, layers) = page_with_layers(&[(
            Some("轰"),
            Some("Rumble"),
            "dev.koharu.text.dialogue",
            "onomatopoeia",
            100.0,
            Some(28.0),
        )]);
        apply_scene_gates(
            &gate_settings(&dictionary),
            &mut session,
            page,
            crate::labels::SourceScript::Chinese,
            Plan::Full,
        );

        let typography = session
            .snapshot()
            .component::<Typography>(layers[0])
            .expect("the layer is in the scene")
            .expect("the fixture set a typography");
        assert_eq!(typography.stroke_width, None, "no halo under a full plan");
        assert_eq!(typography.stroke_color, None);
    }

    /// The duplicate gate follows `plan.letters_text()`: under `Plan::KeepArt`
    /// (which letters text over kept artwork) the redundant half of an
    /// identical-string overlap is refused exactly as under `Plan::Full`, and
    /// under `Plan::CleanOnly` (no text at all) NO gate contributes a refusal
    /// -- both layers carry `SourceText`, so the ungated unread pass has
    /// nothing to add either.
    ///
    /// Proven able to fail INDEPENDENTLY of the `Plan::Full` tests above:
    /// hardcoding gate 4's condition from `settings.skip_duplicate_text &&
    /// plan.letters_text()` to `settings.skip_duplicate_text` turned only the
    /// CleanOnly arm red while all three earlier tests stayed green -- which
    /// is what makes this pair discriminating rather than a re-run of
    /// `a_duplicate_reason_never_overwrites_an_earlier_refusal`.
    #[test]
    fn the_duplicate_gate_follows_the_plans_lettering_choice() {
        let dictionary = crate::sfx::Dictionary::default();
        let specs = [
            (Some("轰轰"), Some("Boom"), "dev.koharu.text.free-text", "text", 100.0, None),
            (Some("轰轰"), Some("Boom"), "dev.koharu.text.free-text", "text", 120.0, None),
        ];

        let (mut session, page, _contents, _layers) = page_with_layers(&specs);
        let refused = apply_scene_gates(
            &gate_settings(&dictionary),
            &mut session,
            page,
            crate::labels::SourceScript::Chinese,
            Plan::KeepArt,
        );
        assert_eq!(
            refused.len(),
            1,
            "KeepArt letters text, so the redundant half is refused: {refused:?}"
        );
        assert!(
            matches!(refused[0].1, crate::labels::Refusal::DuplicateLettering),
            "and for the duplicate gate's own reason: {refused:?}"
        );

        let (mut session, page, _contents, _layers) = page_with_layers(&specs);
        let refused = apply_scene_gates(
            &gate_settings(&dictionary),
            &mut session,
            page,
            crate::labels::SourceScript::Chinese,
            Plan::CleanOnly,
        );
        assert!(
            refused.is_empty(),
            "CleanOnly letters nothing, so no gate refuses: {refused:?}"
        );
    }

    fn guarded() -> Router {
        guarded_with(post(|| async { "translated" }))
    }

    /// Stubs stand in for the real handlers so the route table, the guards and
    /// the panic layer can all be tested without a pipeline behind them.
    fn guarded_with(translate: MethodRouter) -> Router {
        router(
            GuardState {
                token_digest: Some(digest("s3cret")),
                allowed_origins: Vec::new().into(),
                allowed_hosts: vec![HOST.to_owned()].into(),
            },
            Handlers {
                translate,
                status: get(|| async { "status" }),
                warmup: post(|| async { "warmed" }),
                unload: post(|| async { "unloaded" }),
                shutdown: post(|| async { "stopping" }),
                story: post(|| async { "cleared" }),
                glossary: post(|| async { "installed" }),
            },
            1024,
        )
    }

    /// A request with the loopback Host, and the token only when asked for.
    async fn call(method: &str, uri: &str, token: Option<&str>) -> Response {
        let mut request = Request::builder()
            .method(method)
            .uri(uri)
            .header(header::HOST, HOST);
        if let Some(token) = token {
            request = request.header("x-koharu-token", token);
        }
        guarded()
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn health_needs_no_token() {
        let response = guarded()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header(header::HOST, HOST)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn a_cross_origin_post_never_reaches_the_handler() {
        let response = guarded()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/translate")
                    .header(header::HOST, HOST)
                    .header(header::ORIGIN, "https://evil.example")
                    .header("x-koharu-token", "s3cret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn translate_without_the_token_is_plain_text_401() {
        let response = guarded()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/translate")
                    .header(header::HOST, HOST)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/plain; charset=utf-8")
        );
    }

    #[tokio::test]
    async fn translate_is_post_only() {
        let response = guarded()
            .oneshot(
                Request::builder()
                    .uri("/translate")
                    .header(header::HOST, HOST)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[test]
    fn the_busy_message_fits_even_an_absurd_timeout() {
        // --request-timeout-secs takes a u64, so the seconds can be 20 digits.
        let message = queue_timed_out(Duration::from_secs(u64::MAX)).message;
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
    }

    #[tokio::test]
    async fn every_endpoint_but_health_demands_the_token() {
        for (method, uri) in [
            ("POST", "/translate"),
            ("GET", "/status"),
            ("POST", "/warmup"),
            ("POST", "/unload"),
            /* Stop kills the reader's server. Unauthenticated, any page that can
             * reach loopback could end a translation run mid-chapter. */
            ("POST", "/shutdown"),
            ("POST", "/story"),
            /* An unauthenticated glossary write would let any loopback page
             * rewrite what every term in a series is called. */
            ("POST", "/glossary"),
        ] {
            assert_eq!(
                call(method, uri, None).await.status(),
                StatusCode::UNAUTHORIZED,
                "{method} {uri}"
            );
            assert_eq!(
                call(method, uri, Some("wrong")).await.status(),
                StatusCode::UNAUTHORIZED,
                "{method} {uri}"
            );
            assert_eq!(
                call(method, uri, Some("s3cret")).await.status(),
                StatusCode::OK,
                "{method} {uri}"
            );
        }
        // The popup pings this before it has a token to send.
        assert_eq!(call("GET", "/health", None).await.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn the_new_endpoints_reject_the_wrong_method_rather_than_the_caller() {
        // A 401 here would send whoever is holding it wrong wondering about the
        // token instead of the verb.
        assert_eq!(
            call("GET", "/warmup", None).await.status(),
            StatusCode::METHOD_NOT_ALLOWED
        );
        assert_eq!(
            call("POST", "/status", None).await.status(),
            StatusCode::METHOD_NOT_ALLOWED
        );
    }

    #[test]
    fn the_status_body_is_the_shape_the_popup_reads() {
        let status = StatusJson {
            models_loaded: true,
            unload_in_secs: Some(277),
            idle_unload_secs: 300,
            vram: vram_json(&Vram::Known {
                device: "NVIDIA GeForce RTX 5090".to_owned(),
                budget_bytes: 32 * vram::GIB,
                available_bytes: 30 * vram::GIB,
            }),
            sufficient: true,
            needed_bytes: 3_779_571_712,
            provider: models::PROVIDER_OLLAMA,
            model: "qwen3:8b".to_owned(),
            busy: false,
            ocr_substitute: None,
        };
        assert_eq!(
            serde_json::to_value(&status).unwrap(),
            serde_json::json!({
                "models_loaded": true,
                "unload_in_secs": 277,
                "idle_unload_secs": 300,
                "vram": {
                    "budget_bytes": 34_359_738_368_u64,
                    "available_bytes": 32_212_254_720_u64,
                    "device": "NVIDIA GeForce RTX 5090",
                },
                "sufficient": true,
                "needed_bytes": 3_779_571_712_u64,
                "provider": "ollama",
                "model": "qwen3:8b",
                "busy": false,
                "ocr_substitute": null,
            })
        );
    }

    /// Startup state built the way `run` builds it, from a command line.
    fn resolved(extra: &[&str]) -> crate::cli::Resolved {
        use clap::Parser as _;
        let argv = std::iter::once("birelate-server").chain(extra.iter().copied());
        crate::cli::Cli::try_parse_from(argv).unwrap().resolve(None).unwrap()
    }

    /// Through `selection_from`, the function `translate` calls, not just
    /// `models::resolve_ocr`: the wire value and the absent-field fallback are
    /// two arms, and either could skip the substitution on its own.
    #[test]
    fn a_hunyuan_request_is_served_by_the_substitute() {
        use koharu_pipeline::OcrModel;
        let pick = |resolved: &crate::cli::Resolved, ocr: Option<&str>| {
            let form = TranslateForm {
                ocr: ocr.map(str::to_owned),
                ..TranslateForm::default()
            };
            selection_from(&form, &resolved.defaults, &resolved.pinned).unwrap()
        };

        let substituted = resolved(&["--hunyuan-substitute", "paddleocr-vl-1.6"]);
        for ocr in [Some("hunyuan-ocr-1.5"), None] {
            let selection = pick(&substituted, ocr);
            assert_eq!(selection.ocr, OcrModel::PaddleOcrVl1_6, "{ocr:?}");
            // The config that loads, which `verify_applied` holds the stage
            // runner's report to, names the engine that runs.
            let config = desired_config(&selection, &substituted.defaults, &substituted.pinned);
            assert_eq!(models::ocr_model_name(&config.ocr), "paddleocr-vl-1.6");
        }
        // Only Hunyuan is replaced.
        assert_eq!(pick(&substituted, Some("manga-ocr")).ocr, OcrModel::MangaOcr);

        // Control arm: without the flag, Hunyuan is served by Hunyuan.
        let plain = resolved(&[]);
        assert_eq!(pick(&plain, Some("hunyuan-ocr-1.5")).ocr, OcrModel::HunyuanOcr1_5);
        assert_eq!(pick(&plain, None).ocr, OcrModel::HunyuanOcr1_5);
    }

    /// Through `status_json`, the construction `status_of` returns.
    #[test]
    fn status_reports_the_hunyuan_substitute_or_null() {
        let status_with = |extra: &[&str]| {
            let resolved = resolved(extra);
            let headroom = vram::Headroom {
                vram: Vram::Unknown,
                needed_bytes: 0,
                sufficient: true,
            };
            let idle = crate::idle::IdleClock::new(None);
            let status =
                status_json(&resolved.pinned, &resolved.defaults, &idle, &headroom, false, false);
            serde_json::to_value(status).unwrap()
        };
        assert_eq!(
            status_with(&["--hunyuan-substitute", "paddleocr-vl-1.6"])["ocr_substitute"],
            serde_json::json!({ "requested": "hunyuan-ocr-1.5", "served_by": "paddleocr-vl-1.6" })
        );
        let off = status_with(&[]);
        // Present as `null`, not absent: the popup tells the two apart.
        assert!(off.get("ocr_substitute").is_some_and(serde_json::Value::is_null), "{off}");
    }

    #[test]
    fn the_translate_body_keeps_its_old_fields_and_gains_the_miss_report() {
        let payload = TranslateJson {
            image: "data:image/png;base64,AA==".to_owned(),
            regions: Vec::new(),
            ms: 12_345,
            untranslated: vec![3, 7],
            // One measured page lost regions 3, 4, 5 and 6 like this, with
            // `untranslated` empty beside them.
            dropped: vec![3, 4, 5, 6],
            split_fragments: Vec::new(),
            placement_overflow: Vec::new(),
            placement_clamped: Vec::new(),
            translation_pins_applied: 0,
            skipped: Vec::new(),
            truncated: true,
            duplicate_ids: 0,
            out_of_range_ids: 0,
            cut_found: 0,
            still_cut: 0,
            stages: vec![StageTiming {
                stage: "ocr".to_owned(),
                model: "baberu-ocr".to_owned(),
                ms: 2_960,
                // Non-zero so the assertion below proves the field is actually
                // serialized. A zero here would pass against a struct that had
                // silently dropped it, which is the shape of green-but-inert
                // this project keeps finding.
                wait_ms: 120,
            }],
            stages_selected: Vec::new(),
            layout_warnings: Vec::new(),
            rendered_text: Vec::new(),
            story_context: None,
            glossary_terms: None,
            containment_clause: false,
            source_script: "unknown",
            edge_hints: Vec::new(),
        };
        let body = serde_json::to_value(&payload).unwrap();
        // The three the extension already reads, unchanged in name and type.
        assert!(body["image"].is_string(), "{body}");
        assert!(body["regions"].is_array(), "{body}");
        assert_eq!(body["ms"], 12_345);
        assert_eq!(body["untranslated"], serde_json::json!([3, 7]));
        assert_eq!(body["truncated"], true);
        // Beside it, never folded into it: the two describe different failures
        // and `untranslated` structurally cannot hold this one.
        assert_eq!(body["dropped"], serde_json::json!([3, 4, 5, 6]));
        // Named fields rather than a positional tuple: a benchmark joins on
        // `stage`, and a page whose stages get reordered by the scheduler would
        // otherwise silently attribute the LLM's time to OCR.
        assert_eq!(
            body["stages"],
            serde_json::json!([{
                "stage": "ocr",
                "model": "baberu-ocr",
                "ms": 2_960,
                "wait_ms": 120,
            }])
        );
    }

    /// The cut-repair verdict rides the body as two always-present integers:
    /// without them a repaired page and a silently cut one serialize
    /// identically.
    /// Non-zero values, so a struct that silently dropped the fields cannot
    /// pass -- and the zero case asserts PRESENCE, because an absent key and a
    /// zero count must not be conflated by the extension's reader.
    #[test]
    fn a_json_body_carries_the_cut_repair_verdict() {
        let cut = TranslateJson {
            cut_found: 3,
            still_cut: 1,
            ..empty_translate_json()
        };
        let body = serde_json::to_value(&cut).unwrap();
        assert_eq!(body["cut_found"], 3, "{body}");
        assert_eq!(body["still_cut"], 1, "{body}");

        let clean = serde_json::to_value(&empty_translate_json()).unwrap();
        assert_eq!(clean["cut_found"], 0, "an absent key would read as unknown");
        assert_eq!(clean["still_cut"], 0, "an absent key would read as unknown");
    }

    /// One neutral body for tests that assert a single field: every counter
    /// zero, every list empty.
    fn empty_translate_json() -> TranslateJson {
        TranslateJson {
            image: String::new(),
            regions: Vec::new(),
            ms: 0,
            untranslated: Vec::new(),
            dropped: Vec::new(),
            split_fragments: Vec::new(),
            placement_overflow: Vec::new(),
            placement_clamped: Vec::new(),
            translation_pins_applied: 0,
            skipped: Vec::new(),
            truncated: false,
            duplicate_ids: 0,
            out_of_range_ids: 0,
            cut_found: 0,
            still_cut: 0,
            stages: Vec::new(),
            stages_selected: Vec::new(),
            layout_warnings: Vec::new(),
            rendered_text: Vec::new(),
            story_context: None,
            glossary_terms: None,
            containment_clause: false,
            source_script: "unknown",
            edge_hints: Vec::new(),
        }
    }

    /// An empty array, not a missing key: a benchmark omits its per-stage
    /// column when the field is absent, and would silently do so on every page
    /// if a run that reported nothing serialized as `null`.
    #[test]
    fn a_run_that_reported_no_stages_still_carries_the_array() {
        let payload = TranslateJson {
            image: String::new(),
            regions: Vec::new(),
            ms: 0,
            untranslated: Vec::new(),
            dropped: Vec::new(),
            split_fragments: Vec::new(),
            placement_overflow: Vec::new(),
            placement_clamped: Vec::new(),
            translation_pins_applied: 0,
            skipped: Vec::new(),
            truncated: false,
            duplicate_ids: 0,
            out_of_range_ids: 0,
            cut_found: 0,
            still_cut: 0,
            stages: Vec::new(),
            stages_selected: Vec::new(),
            layout_warnings: Vec::new(),
            rendered_text: Vec::new(),
            story_context: None,
            glossary_terms: None,
            containment_clause: false,
            source_script: "unknown",
            edge_hints: Vec::new(),
        };
        let body = serde_json::to_value(&payload).unwrap();
        assert_eq!(body["stages"], serde_json::json!([]));
        assert_eq!(body["stages_selected"], serde_json::json!([]));
        /* Same rule again, and this one is the reason the rule exists. A server
         * too old to check for erased-and-unlettered boxes must not be
         * indistinguishable from one that checked and found none -- 59 measured
         * instances answered with every other counter clean, so absence here is
         * the only thing left that could be read as an all-clear. */
        assert_eq!(body["dropped"], serde_json::json!([]));
        /* Same rule, same reason: the renderer computes a verdict on every layer
         * it sets and `render_png` used to discard it. An absent key would read
         * as "this server does not check", which is exactly the state surfacing
         * it exists to leave behind. */
        assert_eq!(body["layout_warnings"], serde_json::json!([]));
    }

    #[test]
    fn a_layer_the_renderer_could_not_set_is_reported_with_its_numbers() {
        let payload = TranslateJson {
            image: String::new(),
            regions: Vec::new(),
            ms: 0,
            untranslated: Vec::new(),
            dropped: Vec::new(),
            split_fragments: Vec::new(),
            placement_overflow: Vec::new(),
            placement_clamped: Vec::new(),
            translation_pins_applied: 0,
            skipped: Vec::new(),
            truncated: false,
            duplicate_ids: 0,
            out_of_range_ids: 0,
            cut_found: 0,
            still_cut: 0,
            stages: Vec::new(),
            stages_selected: Vec::new(),
            story_context: None,
            glossary_terms: None,
            containment_clause: false,
            source_script: "unknown",
            edge_hints: Vec::new(),
            rendered_text: Vec::new(),
            layout_warnings: vec![
                crate::render::LayoutWarning {
                    kind: "too_small",
                    entity: koharu_scene::EntityId::new(),
                    region: Some(0),
                    font_size: 9.0,
                    minimum_font_size: Some(9.0),
                    actual_width: None,
                    actual_height: None,
                },
                crate::render::LayoutWarning {
                    kind: "overflow",
                    entity: koharu_scene::EntityId::new(),
                    region: Some(3),
                    font_size: 24.0,
                    minimum_font_size: None,
                    actual_width: Some(312.5),
                    actual_height: Some(48.0),
                },
            ],
        };
        let body = serde_json::to_value(&payload).unwrap();
        let warnings = body["layout_warnings"].as_array().unwrap();
        assert_eq!(warnings.len(), 2, "{body}");
        // The kind is the whole of what a reader acts on, and the numbers are
        // what a benchmark would chart, so both have to survive serialization.
        assert_eq!(warnings[0]["kind"], "too_small");
        assert_eq!(warnings[0]["minimum_font_size"], 9.0);
        assert_eq!(warnings[1]["kind"], "overflow");
        assert_eq!(warnings[1]["actual_width"], 312.5);
        // The scene entity is a uuid regenerated every run and means nothing to
        // the extension, so it must never reach the wire -- same rule as
        // RegionOut::content.
        assert!(warnings[0].get("entity").is_none(), "{body}");
        // Absent rather than null, so a `too_small` carries no overflow numbers.
        assert!(warnings[0].get("actual_width").is_none(), "{body}");
        assert!(warnings[1].get("minimum_font_size").is_none(), "{body}");
    }

    /// The placed extent has to reach the wire under its own name, beside the box
    /// rather than instead of it.
    ///
    /// Both rectangles are needed and they answer different questions: `x/y/w/h`
    /// is the box the compositor handed the layer, `placed_*` is where auto-fit
    /// actually set the text inside it. A reader that had only one of them could
    /// not tell a region that fills its box from one centred in a box three times
    /// its size -- and telling those apart is the entire reason this pair exists,
    /// because the second is what made bounding-box overlap useless as a collision
    /// predictor.
    ///
    /// `entity` must NOT reach the wire: it is a uuid regenerated every run, and
    /// `region` is the index a reader can actually act on.
    #[test]
    fn a_solved_layer_reaches_the_wire_with_both_its_rectangles() {
        let payload = TranslateJson {
            image: String::new(),
            regions: Vec::new(),
            ms: 0,
            untranslated: Vec::new(),
            dropped: Vec::new(),
            split_fragments: Vec::new(),
            placement_overflow: Vec::new(),
            placement_clamped: Vec::new(),
            translation_pins_applied: 0,
            skipped: Vec::new(),
            truncated: false,
            duplicate_ids: 0,
            out_of_range_ids: 0,
            cut_found: 0,
            still_cut: 0,
            stages: Vec::new(),
            stages_selected: Vec::new(),
            story_context: None,
            glossary_terms: None,
            containment_clause: false,
            source_script: "unknown",
            edge_hints: Vec::new(),
            layout_warnings: Vec::new(),
            rendered_text: vec![crate::render::RenderedText {
                entity: koharu_scene::EntityId::new(),
                region: Some(2),
                font_size: 21.5,
                chars: 9,
                x: 100.0,
                y: 200.0,
                width: 300.0,
                height: 400.0,
                placed_x: 140.0,
                placed_y: 260.0,
                placed_width: 220.0,
                placed_height: 60.0,
                placed_angle: -12.5,
            }],
        };
        let body = serde_json::to_value(&payload).unwrap();
        let text = body["rendered_text"].as_array().unwrap();
        assert_eq!(text.len(), 1, "{body}");
        assert_eq!(text[0]["region"], 2);
        // The box, unchanged -- every existing reader of this array keeps working.
        assert_eq!(
            (
                &text[0]["x"],
                &text[0]["y"],
                &text[0]["width"],
                &text[0]["height"]
            ),
            (
                &serde_json::json!(100.0),
                &serde_json::json!(200.0),
                &serde_json::json!(300.0),
                &serde_json::json!(400.0)
            ),
            "{body}"
        );
        // The placed extent, which is a different rectangle and must not have been
        // wired to the box by copy-paste.
        assert_eq!(
            (
                &text[0]["placed_x"],
                &text[0]["placed_y"],
                &text[0]["placed_width"],
                &text[0]["placed_height"]
            ),
            (
                &serde_json::json!(140.0),
                &serde_json::json!(260.0),
                &serde_json::json!(220.0),
                &serde_json::json!(60.0)
            ),
            "{body}"
        );
        // The angle rides with the rectangle or the rectangle is wrong on every
        // angled effect, which is the population the whole pair is aimed at.
        assert_eq!(text[0]["placed_angle"], -12.5, "{body}");
        assert!(text[0].get("entity").is_none(), "{body}");
    }

    /// Absent would read as "this server does not check", which is the state the
    /// whole signal exists to leave behind.
    #[test]
    fn a_fully_translated_page_still_says_so_explicitly() {
        let payload = TranslateJson {
            image: String::new(),
            regions: Vec::new(),
            ms: 0,
            untranslated: Vec::new(),
            dropped: Vec::new(),
            split_fragments: Vec::new(),
            placement_overflow: Vec::new(),
            placement_clamped: Vec::new(),
            translation_pins_applied: 0,
            skipped: Vec::new(),
            truncated: false,
            duplicate_ids: 0,
            out_of_range_ids: 0,
            cut_found: 0,
            still_cut: 0,
            stages: Vec::new(),
            stages_selected: Vec::new(),
            layout_warnings: Vec::new(),
            rendered_text: Vec::new(),
            story_context: None,
            glossary_terms: None,
            containment_clause: false,
            source_script: "unknown",
            edge_hints: Vec::new(),
        };
        let body = serde_json::to_value(&payload).unwrap();
        assert_eq!(body["untranslated"], serde_json::json!([]));
        assert_eq!(body["truncated"], false);
        // Same rule, same reason: absent would read as "this server does not
        // count them", which is the state this pair exists to leave behind.
        assert_eq!(body["duplicate_ids"], 0);
        assert_eq!(body["out_of_range_ids"], 0);
    }

    /// The shape `untranslated` alone cannot describe, pinned as a wire contract.
    ///
    /// Both bodies below report the same seven regions in the source language.
    /// One model stopped early; the other answered the page and addressed seven
    /// of its replies to ids it had already used. Those want opposite fixes --
    /// a bigger token budget, or none at all -- and before these two keys the two
    /// bodies were byte-identical.
    #[test]
    fn a_mis_addressed_reply_is_distinguishable_from_one_that_stopped_early() {
        let base = || TranslateJson {
            image: String::new(),
            regions: Vec::new(),
            ms: 0,
            untranslated: vec![1, 2, 3, 4, 5, 6, 7],
            dropped: Vec::new(),
            split_fragments: Vec::new(),
            placement_overflow: Vec::new(),
            placement_clamped: Vec::new(),
            translation_pins_applied: 0,
            skipped: Vec::new(),
            truncated: false,
            duplicate_ids: 0,
            out_of_range_ids: 0,
            cut_found: 0,
            still_cut: 0,
            stages: Vec::new(),
            stages_selected: Vec::new(),
            layout_warnings: Vec::new(),
            rendered_text: Vec::new(),
            story_context: None,
            glossary_terms: None,
            containment_clause: false,
            source_script: "unknown",
            edge_hints: Vec::new(),
        };

        let stopped_early = TranslateJson {
            truncated: true,
            ..base()
        };
        let mis_addressed = TranslateJson {
            duplicate_ids: 7,
            out_of_range_ids: 2,
            ..base()
        };

        let stopped = serde_json::to_value(&stopped_early).unwrap();
        let addressed = serde_json::to_value(&mis_addressed).unwrap();
        assert_eq!(stopped["untranslated"], addressed["untranslated"]);
        assert_eq!(stopped["duplicate_ids"], 0);
        assert_eq!(stopped["out_of_range_ids"], 0);
        assert_eq!(addressed["duplicate_ids"], 7);
        assert_eq!(addressed["out_of_range_ids"], 2);
        // The whole point: the two bodies are no longer the same page.
        assert_ne!(stopped, addressed);
    }

    /// `Operation::stages` is `pub(crate)` in koharu, so the resolution itself
    /// cannot be asserted from here -- which is the whole reason the server
    /// reports what the pipeline said rather than what the request asked for.
    /// What is checkable is that the operation we build names no inpainting.
    #[test]
    fn the_keep_the_art_operation_names_no_inpainting() {
        let Operation::Stages { stages } = keep_the_art() else {
            panic!("the keep-the-art operation should be an explicit stage list");
        };
        assert!(!stages.contains(&Stage::Inpainting));
        assert!(stages.contains(&Stage::Detection));
        assert!(stages.contains(&Stage::Ocr));
        // Translation has to be in the list. Koharu enforces prerequisites only
        // among the stages actually selected, so dropping it would not error --
        // it would render the page with `fallback_to_source_text` re-typesetting
        // the Japanese over itself, and answer 200.
        assert!(stages.contains(&Stage::Translation));

        /* The spellings are a wire contract, not a detail. Both the header and
         * `stages_selected` are produced by `Stage`'s `Display`, and the
         * extension decides whether the server honoured the switch by looking
         * for this exact word -- a drift to "Inpainting" would read as "the art
         * survived" on every erased page. */
        assert_eq!(Stage::Detection.to_string(), "detection");
        assert_eq!(Stage::Ocr.to_string(), "ocr");
        assert_eq!(Stage::Translation.to_string(), "translation");
        assert_eq!(Stage::Inpainting.to_string(), "inpainting");
    }

    /// The diagnostic's whole value is that the page comes back *blank*. Asking
    /// for OCR as well would put `SourceText` in the scene, and the renderer's
    /// `fallback_to_source_text` would then re-typeset the Japanese over the
    /// artwork -- a page that looks like a broken translation rather than a
    /// clean plate, and the exact trap the operation is spelled to avoid.
    #[test]
    fn cleaning_a_page_asks_for_no_ocr_so_nothing_is_typeset_back() {
        let Operation::Through { stage } = Plan::CleanOnly.operation() else {
            panic!("clean-only should be a Through operation");
        };
        assert_eq!(stage, Stage::Inpainting);
        assert!(!Plan::CleanOnly.letters_text());
        assert!(!Plan::CleanOnly.kept_art());
    }

    #[test]
    fn each_plan_asks_for_a_different_thing() {
        assert_eq!(Plan::Full.operation(), Operation::Full);
        assert!(Plan::Full.letters_text() && !Plan::Full.kept_art());
        // `KeepArt` is the only plan that needs the in-bubble halo: the only one
        // whose balloons still have Japanese under text it actually letters.
        // `DetectOnly` also keeps the art, but letters nothing, so the halo
        // question never arises there.
        assert!(Plan::KeepArt.kept_art() && Plan::KeepArt.letters_text());
        assert_ne!(Plan::KeepArt.operation(), Operation::Full);
    }

    /// **Detection and nothing else, for the seam lookahead.**
    ///
    /// The seam plan is computed from GEOMETRY ALONE: `extension/seam.js`'s
    /// `seamEdges` reads `seamEffectiveBox` — x/y/width/height — and never touches
    /// `source` or `translated`. Both `regions` and `edge_hints` come out of
    /// detection, so a lookahead that decides every boundary in a strip before the
    /// reader reaches it needs this stage and no other.
    ///
    /// **Why the cheap alternatives are not cheap.** `clean_only` adds inpainting
    /// (~617 ms) and hands back a cleaned page; `skip_inpainting` adds OCR
    /// (~442 ms) and translation (~1,275 ms). Detection is ~119 ms, so the nearest
    /// existing flag costs 6–16x what the pre-pass needs.
    ///
    /// **Spelled as an explicit list, not `Through { Detection }`**, for the reason
    /// [`keep_the_art`] gives: the list says what it means and cannot quietly
    /// change meaning upstream.
    #[test]
    fn detecting_only_asks_for_detection_and_nothing_else() {
        let Operation::Stages { stages } = Plan::DetectOnly.operation() else {
            panic!("detect-only should be an explicit stage list");
        };
        assert_eq!(stages, vec![Stage::Detection]);
        /* No OCR, so no `SourceText` in the scene, so the renderer's
         * `fallback_to_source_text` has nothing to re-typeset -- the same property
         * `CleanOnly` depends on, and the trap `form.rs` warns an arbitrary stage
         * subset walks into. */
        assert!(!stages.contains(&Stage::Ocr));
        assert!(!Plan::DetectOnly.letters_text());
        // No inpainting stage, so the balloons keep their pixels. Inert -- nothing
        // is lettered under this plan -- but true, and a false answer here would
        // be a lie waiting for the first caller that letters anything.
        assert!(Plan::DetectOnly.kept_art());
        assert_ne!(Plan::DetectOnly.operation(), Operation::Full);
    }

    /// **The three plan flags are mutually exclusive, and that is now testable.**
    ///
    /// The exclusion used to live inline in the handler, where nothing could reach
    /// it: a request is needed to exercise a handler, so the one guard that turns
    /// a contradictory request into a 400 had no test at all. [`choose_plan`] is
    /// the predicate the handler calls, which is the level this has to be asserted
    /// at — two halves of an `||` tested separately do not test the join.
    #[test]
    fn contradictory_plan_flags_are_refused_and_each_one_alone_is_honoured() {
        assert_eq!(choose_plan(false, false, false).unwrap(), Plan::Full);
        assert_eq!(choose_plan(true, false, false).unwrap(), Plan::KeepArt);
        assert_eq!(choose_plan(false, true, false).unwrap(), Plan::CleanOnly);
        assert_eq!(choose_plan(false, false, true).unwrap(), Plan::DetectOnly);
        // Every pair, and the triple. None of them means anything.
        for (a, b, c) in [
            (true, true, false),
            (true, false, true),
            (false, true, true),
            (true, true, true),
        ] {
            assert!(
                choose_plan(a, b, c).is_err(),
                "({a}, {b}, {c}) should be refused"
            );
        }
    }

    /// The retry's seed at the edge. The dangerous arm is the REFUSAL: a
    /// garbled seed read as "absent" keeps the fixed sampler constant and
    /// answers the retry with the byte-identical page it was asked to replace
    /// -- a no-op with no symptom. Proven able to fail by making
    /// `parse_seed` swallow garbage as `None`: the refusal arm went red at
    /// once.
    /// The box editor's edge parse. The dangerous arms are the SILENT ones: a
    /// half-parsed removal leaves a deleted box erasing artwork, and a
    /// skimmed addition drops the reader's correction. Proven able to fail by
    /// making garbage parse as an empty list: the refusal asserts went red at
    /// once.
    #[test]
    fn caller_region_lists_parse_exactly_or_are_refused_never_skimmed() {
        assert_eq!(parse_caller_regions("regions_add", None).unwrap(), vec![]);
        assert_eq!(
            parse_caller_regions("regions_add", Some("  ")).unwrap(),
            vec![]
        );
        assert_eq!(parse_caller_regions("regions_add", Some("[]")).unwrap(), vec![]);
        let one = parse_caller_regions(
            "regions_add",
            Some(r#"[{"x": 10.5, "y": 20.0, "width": 100.0, "height": 40.0}]"#,),
        )
        .unwrap();
        assert_eq!(
            one,
            vec![koharu_pipeline::CallerRegion {
                x: 10.5,
                y: 20.0,
                width: 100.0,
                height: 40.0
            }]
        );
        for garbage in [
            "junk",
            r#"[{"x": 1}]"#,
            r#"[{"x": 1, "y": 2, "widht": 3, "height": 4}]"#,
            r#"[{"x": 1, "y": 2, "width": 0, "height": 4}]"#,
            r#"[{"x": 1, "y": 2, "width": 3, "height": -4}]"#,
        ] {
            assert!(
                parse_caller_regions("regions_remove", Some(garbage)).is_err(),
                "{garbage:?} must be refused, never skimmed"
            );
        }
        let over_cap = format!(
            "[{}]",
            std::iter::repeat(r#"{"x":1,"y":2,"width":3,"height":4}"#)
                .take(CALLER_REGION_CAP + 1)
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(parse_caller_regions("regions_add", Some(&over_cap)).is_err());
    }

    /// The placement list's edge parse, refused-never-skimmed for the sharpest
    /// reason in the family: a half-parsed placement letters English somewhere
    /// the reader did not choose, and nothing on the wire says so. Proven able
    /// to fail by making the JSON-error arm answer `Ok(Vec::new())`: the
    /// garbage asserts went red at once.
    #[test]
    fn a_placement_list_parses_exactly_or_is_refused_never_skimmed() {
        assert!(parse_caller_placements(None).unwrap().is_empty());
        assert!(parse_caller_placements(Some("  ")).unwrap().is_empty());
        assert!(parse_caller_placements(Some("[]")).unwrap().is_empty());
        let good = r#"[{"x":410,"y":120,"width":180,"height":240,
                        "place_x":430,"place_y":300,"place_width":300,"place_height":160}]"#;
        let one = parse_caller_placements(Some(good)).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].target.x, 410.0);
        assert_eq!(one[0].target.height, 240.0);
        assert_eq!(one[0].place.x, 430.0);
        assert_eq!(one[0].place.width, 300.0);
        for garbage in [
            "junk",
            r#"[{"x": 1}]"#,
            // A misspelled key must be a 400 today, not a box of height 0 forever.
            r#"[{"x":1,"y":2,"width":3,"height":4,"plaec_x":5,"place_y":6,"place_width":30,"place_height":30}]"#,
            r#"[{"x":1,"y":2,"width":3,"height":4,"place_x":null,"place_y":6,"place_width":30,"place_height":30}]"#,
            // A zero-height TARGET can match nothing on purpose or by accident;
            // refused so the two stay distinguishable.
            r#"[{"x":1,"y":2,"width":3,"height":0,"place_x":5,"place_y":6,"place_width":30,"place_height":30}]"#,
        ] {
            assert!(
                parse_caller_placements(Some(garbage)).is_err(),
                "{garbage:?} must be refused, never skimmed"
            );
        }
        // Under the minimum span the renderer's inset leaves no room at all;
        // the refusal names the constant so the editor mirrors it.
        let tiny = r#"[{"x":1,"y":2,"width":3,"height":4,
                        "place_x":5,"place_y":6,"place_width":20,"place_height":160}]"#;
        let refusal = parse_caller_placements(Some(tiny)).unwrap_err();
        assert!(
            format!("{refusal:?}").contains("24"),
            "the refusal names the minimum: {refusal:?}"
        );
        let over_cap = format!(
            "[{}]",
            std::iter::repeat(
                r#"{"x":1,"y":2,"width":3,"height":4,"place_x":5,"place_y":6,"place_width":30,"place_height":30}"#
            )
            .take(CALLER_REGION_CAP + 1)
            .collect::<Vec<_>>()
            .join(",")
        );
        assert!(parse_caller_placements(Some(&over_cap)).is_err());
    }

    /// A wire-shaped placement for the scene tests below.
    fn caller_placement(
        target: (f64, f64, f64, f64),
        place: (f64, f64, f64, f64),
    ) -> CallerPlacement {
        CallerPlacement {
            target: koharu_pipeline::CallerRegion {
                x: target.0 as f32,
                y: target.1 as f32,
                width: target.2 as f32,
                height: target.3 as f32,
            },
            place: koharu_pipeline::CallerRegion {
                x: place.0 as f32,
                y: place.1 as f32,
                width: place.2 as f32,
                height: place.3 as f32,
            },
        }
    }

    /// The observable is `regions()`'s own `fit_*`, never `edit.set`'s `Ok` --
    /// asserting the write succeeded would stay green with the pass unwired
    /// from what the wire actually reports. Red against the absent module.
    #[test]
    fn a_caller_placement_becomes_the_layers_own_fit_box() {
        let (mut session, page, contents, _layers) = page_with_layers(&[(
            Some("こんにちは"),
            Some("Hello"),
            "dev.koharu.text.free-text",
            "text",
            100.0,
            None,
        )]);
        let before = crate::regions::regions(&session.snapshot(), page);
        let region = before
            .iter()
            .find(|region| region.content == contents[0])
            .expect("the fixture region is on the wire");
        let placed = crate::placement::apply_caller_placement(
            &mut session,
            page,
            &[caller_placement(
                (region.x, region.y, region.width, region.height),
                (400.0, 500.0, 300.0, 160.0),
            )],
        );
        assert_eq!(placed, 1);
        let after = crate::regions::regions(&session.snapshot(), page);
        let region = after
            .iter()
            .find(|region| region.content == contents[0])
            .expect("still on the wire");
        assert_eq!(
            (region.fit_x, region.fit_y, region.fit_width, region.fit_height),
            (Some(400.0), Some(500.0), Some(300.0), Some(160.0)),
            "the caller's rectangle IS the layer's fit box"
        );
    }

    /// Two arms in one test, and the second is the one that matters: it is
    /// what stops "inside" quietly becoming "intersects".
    #[test]
    fn a_placement_targets_by_centre_inside_not_by_equality() {
        let (mut session, page, contents, _layers) = page_with_layers(&[(
            Some("こんにちは"),
            Some("Hello"),
            "dev.koharu.text.free-text",
            "text",
            100.0,
            None,
        )]);
        let before = crate::regions::regions(&session.snapshot(), page);
        let region = &before[0];
        // 10px looser on every side than the region: equality would miss it.
        let placed = crate::placement::apply_caller_placement(
            &mut session,
            page,
            &[caller_placement(
                (region.x - 10.0, region.y - 10.0, region.width + 20.0, region.height + 20.0),
                (400.0, 500.0, 300.0, 160.0),
            )],
        );
        assert_eq!(placed, 1, "a loose target still lands by its centre");

        // Overlapping the region's corner, with the centre OUTSIDE: must not
        // land, and the fit box must keep the value the placement above set.
        let placed = crate::placement::apply_caller_placement(
            &mut session,
            page,
            &[caller_placement(
                (region.x - 60.0, region.y - 60.0, 70.0, 70.0),
                (10.0, 10.0, 100.0, 100.0),
            )],
        );
        assert_eq!(placed, 0, "an overlap whose centre is elsewhere is not a match");
        let after = crate::regions::regions(&session.snapshot(), page);
        let region = after
            .iter()
            .find(|region| region.content == contents[0])
            .expect("still on the wire");
        assert_eq!(region.fit_x, Some(400.0), "the non-match wrote nothing");
    }

    /// The counterpart of `admit_caller_boxes`' refusal logging: a target that
    /// matched nothing must be visible as zero, never as success.
    #[test]
    fn a_placement_matching_no_region_writes_nothing_and_returns_zero() {
        let (mut session, page, contents, _layers) = page_with_layers(&[(
            Some("こんにちは"),
            Some("Hello"),
            "dev.koharu.text.free-text",
            "text",
            100.0,
            None,
        )]);
        let before = crate::regions::regions(&session.snapshot(), page);
        let fit_before = before[0].fit_x;
        let placed = crate::placement::apply_caller_placement(
            &mut session,
            page,
            &[caller_placement((5000.0, 5000.0, 40.0, 40.0), (400.0, 500.0, 300.0, 160.0))],
        );
        assert_eq!(placed, 0);
        let after = crate::regions::regions(&session.snapshot(), page);
        let region = after
            .iter()
            .find(|region| region.content == contents[0])
            .expect("still on the wire");
        assert_eq!(region.fit_x, fit_before, "no layer was touched");
    }

    /// Pins the never-rotate rule structurally: the written frame is an
    /// axis-aligned rectangle, so nothing on this path can ever rotate the
    /// lettered English.
    #[test]
    fn a_caller_placement_sets_no_angle() {
        let (mut session, page, _contents, layers) = page_with_layers(&[(
            Some("こんにちは"),
            Some("Hello"),
            "dev.koharu.text.free-text",
            "text",
            100.0,
            None,
        )]);
        let before = crate::regions::regions(&session.snapshot(), page);
        let region = &before[0];
        crate::placement::apply_caller_placement(
            &mut session,
            page,
            &[caller_placement(
                (region.x, region.y, region.width, region.height),
                (400.0, 500.0, 300.0, 160.0),
            )],
        );
        let snapshot = session.snapshot();
        let frame = snapshot
            .text_layer(layers[0])
            .expect("the fixture layer")
            .frame()
            .expect("frame readable")
            .expect("frame present");
        let points = &frame.points;
        assert_eq!(points.len(), 4);
        assert_eq!(points[0].y, points[1].y, "top edge is level");
        assert_eq!(points[1].x, points[2].x, "right edge is plumb");
        assert_eq!(points[2].y, points[3].y, "bottom edge is level");
        assert_eq!(points[3].x, points[0].x, "left edge is plumb");
    }

    /// The wiring test, and the reason `finish_scene` exists at all: with
    /// `apply_caller_placement` written but never called, every test above
    /// stays green while the feature ships unwired. Proven able to fail by
    /// exactly that cut.
    #[test]
    fn finish_scene_places_the_boxes_the_gates_left_behind() {
        let dictionary = crate::sfx::Dictionary::default();
        let (mut session, page, contents, _layers) = page_with_layers(&[(
            Some("こんにちは"),
            Some("Hello"),
            "dev.koharu.text.free-text",
            "text",
            100.0,
            None,
        )]);
        let before = crate::regions::regions(&session.snapshot(), page);
        let region = &before[0];
        let placements = [caller_placement(
            (region.x, region.y, region.width, region.height),
            (400.0, 500.0, 300.0, 160.0),
        )];
        let (refused, placed, pinned) = finish_scene(
            &gate_settings(&dictionary),
            &mut session,
            page,
            crate::labels::SourceScript::Japanese,
            Plan::Full,
            &placements,
            &[],
            &[],
        );
        assert_eq!(pinned, 0, "no pins were sent");
        assert!(refused.is_empty(), "ordinary dialogue survives the gates: {refused:?}");
        assert_eq!(placed, 1);
        let after = crate::regions::regions(&session.snapshot(), page);
        let region = after
            .iter()
            .find(|region| region.content == contents[0])
            .expect("still on the wire");
        assert_eq!(region.fit_x, Some(400.0), "the placement ran through the seam");
    }

    /// The pin list's edge parse. The stake this family's discipline protects
    /// here: a skimmed pin silently re-rolls a bubble the reader believed
    /// they were keeping. Proven able to fail by making the JSON-error arm
    /// answer `Ok(Vec::new())`.
    #[test]
    fn a_translation_pin_list_parses_exactly_or_is_refused_never_skimmed() {
        assert!(parse_translation_pins(None).unwrap().is_empty());
        assert!(parse_translation_pins(Some("  ")).unwrap().is_empty());
        assert!(parse_translation_pins(Some("[]")).unwrap().is_empty());
        let one = parse_translation_pins(Some(r#"[{"s":"カサカサ…","t":"AH..."}]"#)).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].source, "カサカサ…");
        assert_eq!(one[0].translation, "AH...");
        for garbage in [
            "junk",
            r#"[{"s":"a"}]"#,
            r#"[{"s":"a","t":"b","u":"c"}]"#,
            r#"[{"s":"","t":"b"}]"#,
            r#"[{"s":"a","t":"  "}]"#,
        ] {
            assert!(
                parse_translation_pins(Some(garbage)).is_err(),
                "{garbage:?} must be refused, never skimmed"
            );
        }
        let over_cap = format!(
            "[{}]",
            std::iter::repeat(r#"{"s":"a","t":"b"}"#)
                .take(TRANSLATION_PIN_CAP + 1)
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(parse_translation_pins(Some(&over_cap)).is_err());
    }

    /// The observable is `regions()`'s own `translated`, never `edit.set`'s
    /// `Ok` -- the placement tests' rule. Byte-identity both ways: the match
    /// lands on the exact string and refuses a whisker off, because a fuzzy
    /// pin would glue a stale translation to a genuinely different read.
    #[test]
    fn a_pin_keeps_a_regions_wording_and_matches_by_byte_identity() {
        let (mut session, page, contents, _layers) = page_with_layers(&[
            (Some("カサカサ…"), Some("THIS IS..."), "dev.koharu.text.free-text", "text", 100.0, None),
            (Some("ごめんなさい"), Some("I'M SORRY."), "dev.koharu.text.free-text", "text", 400.0, None),
        ]);
        let pins = [
            crate::routes::TranslationPin {
                source: "カサカサ…".to_owned(),
                translation: "AH...".to_owned(),
            },
            // A whisker off the second region's source: must match nothing.
            crate::routes::TranslationPin {
                source: "ごめんなさい ".to_owned(),
                translation: "WRONG".to_owned(),
            },
        ];
        let pinned = crate::pins::apply_translation_pins(&mut session, page, &pins);
        assert_eq!(pinned, 1, "one exact match, one refused whisker");
        let regions = crate::regions::regions(&session.snapshot(), page);
        let first = regions.iter().find(|r| r.content == contents[0]).unwrap();
        assert_eq!(first.translated, "AH...", "the stored wording is kept verbatim");
        let second = regions.iter().find(|r| r.content == contents[1]).unwrap();
        assert_eq!(
            second.translated, "I'M SORRY.",
            "a near-miss pins nothing -- the fresh draw stands"
        );
    }

    /// The wiring test, `finish_scene`'s second: with the pass written but
    /// never called, the test above stays green while every apply re-rolls.
    /// Proven able to fail by exactly that cut.
    #[test]
    fn finish_scene_pins_the_wording_before_the_gates_judge() {
        let dictionary = crate::sfx::Dictionary::default();
        let (mut session, page, contents, _layers) = page_with_layers(&[(
            Some("カサカサ…"),
            Some("THIS IS..."),
            "dev.koharu.text.free-text",
            "text",
            100.0,
            None,
        )]);
        let pins = [crate::routes::TranslationPin {
            source: "カサカサ…".to_owned(),
            translation: "AH...".to_owned(),
        }];
        let (refused, _placed, pinned) = finish_scene(
            &gate_settings(&dictionary),
            &mut session,
            page,
            crate::labels::SourceScript::Japanese,
            Plan::Full,
            &[],
            &pins,
            &[],
        );
        assert!(refused.is_empty(), "ordinary text survives the gates: {refused:?}");
        assert_eq!(pinned, 1);
        let regions = crate::regions::regions(&session.snapshot(), page);
        let region = regions.iter().find(|r| r.content == contents[0]).unwrap();
        assert_eq!(region.translated, "AH...", "the pin ran through the seam");
    }

    /// A caller-touched region's layer gains its own rect as a
    /// reader-authored frame. The fixture's layers already carry a frame that
    /// EQUALS the region rect, so both are first doctored to a sentinel --
    /// otherwise the pass writing the region rect back is unobservable and
    /// this test could never go red. Red against the absent pass.
    #[test]
    fn a_caller_touched_region_gains_its_own_frame() {
        let (mut session, page, contents, layers) = page_with_layers(&[
            (Some("カサカサ…"), Some("AH..."), "dev.koharu.text.free-text", "text", 100.0, None),
            (Some("ごめん"), Some("SORRY."), "dev.koharu.text.free-text", "text", 400.0, None),
        ]);
        let sentinel = session
            .snapshot()
            .patch(|edit| {
                edit.set(layers[0], &Geometry::rectangle(1.0, 1.0, 30.0, 30.0))?;
                edit.set(layers[1], &Geometry::rectangle(2.0, 2.0, 40.0, 40.0))?;
                Ok(())
            })
            .expect("the sentinel frames are valid");
        session.commit(sentinel).expect("the sentinel frames commit");
        let before = crate::regions::regions(&session.snapshot(), page);
        let touched = before.iter().find(|r| r.content == contents[0]).unwrap();
        let framed = crate::placement::assert_caller_frames(
            &mut session,
            page,
            &[koharu_pipeline::CallerRegion {
                x: touched.x as f32 - 5.0,
                y: touched.y as f32 - 5.0,
                width: touched.width as f32 + 10.0,
                height: touched.height as f32 + 10.0,
            }],
        );
        assert_eq!(framed, 1, "only the touched region is framed");
        let after = crate::regions::regions(&session.snapshot(), page);
        let touched_after = after.iter().find(|r| r.content == contents[0]).unwrap();
        assert_eq!(
            (touched_after.fit_x, touched_after.fit_width),
            (Some(touched.x), Some(touched.width)),
            "the frame is the region's OWN rect, replacing the sentinel"
        );
        let untouched = after.iter().find(|r| r.content == contents[1]).unwrap();
        assert_eq!(
            (untouched.fit_x, untouched.fit_width),
            (Some(2.0), Some(40.0)),
            "an untouched region keeps its sentinel frame"
        );
    }

    /// The WIRING half, through `finish_scene` -- the direct-call test above
    /// stays green with the pass written and never called. Proven red by
    /// exactly that cut.
    #[test]
    fn finish_scene_frames_the_callers_boxes() {
        let dictionary = crate::sfx::Dictionary::default();
        let (mut session, page, contents, layers) = page_with_layers(&[(
            Some("カサカサ…"),
            Some("AH..."),
            "dev.koharu.text.free-text",
            "text",
            100.0,
            None,
        )]);
        let sentinel = session
            .snapshot()
            .patch(|edit| edit.set(layers[0], &Geometry::rectangle(1.0, 1.0, 30.0, 30.0)))
            .expect("the sentinel frame is valid");
        session.commit(sentinel).expect("the sentinel frame commits");
        let before = crate::regions::regions(&session.snapshot(), page);
        let region = before.iter().find(|r| r.content == contents[0]).unwrap();
        let added = [koharu_pipeline::CallerRegion {
            x: region.x as f32,
            y: region.y as f32,
            width: region.width as f32,
            height: region.height as f32,
        }];
        finish_scene(
            &gate_settings(&dictionary),
            &mut session,
            page,
            crate::labels::SourceScript::Japanese,
            Plan::Full,
            &[],
            &[],
            &added,
        );
        let after = crate::regions::regions(&session.snapshot(), page);
        let framed = after.iter().find(|r| r.content == contents[0]).unwrap();
        assert_eq!(
            (framed.fit_x, framed.fit_width),
            (Some(region.x), Some(region.width)),
            "the frame ran through the seam"
        );
    }

    /// `TranslateJson::of` family. The `dropped` doc above records why this
    /// shape of test exists: a mutation check once cut a field's initialiser to
    /// `Vec::new()` and the suite stayed green while every response shipped an
    /// empty list forever. Proven able to fail by hardcoding
    /// `placement_overflow: Vec::new()` in `of`.
    #[test]
    fn a_json_body_carries_the_placement_overflow_it_found() {
        let mut outcome = outcome_with(Vec::new());
        outcome.placement_overflow = vec![1, 3];
        let payload = TranslateJson::of(outcome, 7);
        assert_eq!(payload.placement_overflow, vec![1, 3]);
        let empty = TranslateJson::of(outcome_with(Vec::new()), 7);
        assert!(
            empty.placement_overflow.is_empty(),
            "always present: empty means checked, never means unreported"
        );
    }

    /// The placement-clamp sibling, same shape and same reason: a field that can be
    /// assigned can be assigned to `Vec::new()`, and then the body says "the
    /// server checked and every box was on the page" forever.
    ///
    /// The second assertion is the one that separates this from
    /// `placement_overflow`: the two lists index DIFFERENT arrays, so a
    /// reader that swaps them gets plausible-looking nonsense rather than an
    /// error. Both are carried, and neither is derived from the other.
    #[test]
    fn a_json_body_carries_the_placements_it_had_to_park_on_the_page() {
        let mut outcome = outcome_with(Vec::new());
        outcome.placement_clamped = vec![0, 2];
        outcome.placement_overflow = vec![4];
        let payload = TranslateJson::of(outcome, 7);
        assert_eq!(payload.placement_clamped, vec![0, 2]);
        assert_eq!(
            payload.placement_overflow,
            vec![4],
            "the neighbour is untouched -- these index different arrays"
        );
        let empty = TranslateJson::of(outcome_with(Vec::new()), 7);
        assert!(
            empty.placement_clamped.is_empty(),
            "always present: empty means checked, never means unreported"
        );
        let body = serde_json::to_value(&empty).unwrap();
        assert_eq!(
            body["placement_clamped"],
            serde_json::json!([]),
            "an absent key would read as an older server, not as a clean page"
        );
    }

    #[test]
    fn a_seed_parses_exactly_or_is_refused_never_defaulted() {
        assert_eq!(parse_seed(None).unwrap(), None);
        assert_eq!(parse_seed(Some("0")).unwrap(), Some(0));
        assert_eq!(parse_seed(Some(" 305419896 ")).unwrap(), Some(305_419_896));
        assert_eq!(parse_seed(Some("4294967295")).unwrap(), Some(u32::MAX));
        for garbage in ["", "junk", "-1", "1.5", "4294967296", "0x10"] {
            assert!(
                parse_seed(Some(garbage)).is_err(),
                "{garbage:?} must be refused, never read as absent"
            );
        }
    }

    fn misses(missed: usize, segments: usize, truncated: bool) -> Misses {
        Misses {
            entities: (0..missed).map(|_| EntityId::new()).collect(),
            segments,
            truncated,
            duplicate_ids: 0,
            out_of_range_ids: 0,
            cut_found: 0,
            still_cut: 0,
        }
    }

    /// The mis-addressed arm, kept apart from `misses` so its six callers stay
    /// spelled as the page shapes they were written for.
    fn mis_addressed(duplicate_ids: usize, out_of_range_ids: usize) -> Misses {
        Misses {
            duplicate_ids,
            out_of_range_ids,
            ..misses(0, 12, false)
        }
    }

    #[test]
    fn only_a_wholly_untranslated_page_is_refused() {
        // A run that reported nothing, and a page with some misses, are both
        // pages worth showing.
        assert!(!Misses::default().nothing_translated());
        assert!(!misses(3, 41, true).nothing_translated());
        assert!(!misses(40, 41, true).nothing_translated());
        // Nothing translated at all is not a partial page; it is the source text
        // laid out in a translation font.
        assert!(misses(41, 41, true).nothing_translated());
    }

    #[test]
    fn the_refusal_names_the_count_and_the_cause_inside_the_window() {
        let error = nothing_translated(&misses(41, 41, true));
        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert!(error.message.contains("all 41 regions"), "{}", error.message);
        assert!(error.message.contains("token cap"), "{}", error.message);
        assert!(error.message.len() <= BUDGET, "{}", error.message);

        // A remote provider cannot report a finish reason, so the sentence must
        // still stand without one.
        let quiet = nothing_translated(&misses(9, 9, false));
        assert!(!quiet.message.contains("token cap"), "{}", quiet.message);
        assert!(quiet.message.len() <= BUDGET, "{}", quiet.message);

        let absurd = nothing_translated(&misses(0, usize::MAX, true));
        assert!(absurd.message.len() <= BUDGET, "{}", absurd.message);
    }

    #[test]
    fn the_png_reply_only_grows_headers_when_there_is_a_miss_to_report() {
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
        let clean = headers.clone();
        untranslated_headers(&mut headers, &misses(3, 41, false), 0, 41);
        assert_eq!(headers["x-birelate-untranslated"], "3/41");
        assert_eq!(headers["x-birelate-truncated"], "false");
        assert_eq!(headers[header::CONTENT_TYPE], "image/png");
        // And the untouched map is exactly what the server has always sent.
        assert_eq!(clean.len(), 1);
    }

    /// The shape this gate exists for. A reply cut off *inside* the last
    /// segment's text is repaired into a complete entry, so every id is
    /// answered and `entities` is empty -- while that bubble ends mid-word.
    /// Gating on the entity list alone called such a page clean.
    #[test]
    fn a_truncation_that_missed_no_ids_is_still_worth_reporting() {
        assert!(misses(0, 12, true).worth_reporting());
        assert!(misses(3, 41, false).worth_reporting());
        assert!(misses(3, 41, true).worth_reporting());
        // A run that reported nothing still adds nothing, so an ordinary page
        // answers header for header as it always has.
        assert!(!Misses::default().worth_reporting());

        // It is a partial page, not a refusal: `nothing_translated` counts
        // entities, and there are none.
        assert!(!misses(0, 12, true).nothing_translated());

        let mut headers = HeaderMap::new();
        untranslated_headers(&mut headers, &misses(0, 12, true), 0, 12);
        assert_eq!(headers["x-birelate-untranslated"], "0/12");
        assert_eq!(headers["x-birelate-truncated"], "true");
        // Nothing was mis-addressed, so nothing is said about it -- an ordinary
        // truncated page answers header for header as it always has.
        assert!(headers.get("x-birelate-dropped-ids").is_none());
    }

    /// The third shape of this event, and the reason it no longer implies a
    /// token cap: a provider free to answer more entries than it was asked for
    /// can name one id twice and still cover every one. `entities` is empty,
    /// `truncated` is false, and the two headers above would call it clean.
    #[test]
    fn a_reply_that_only_mis_addressed_its_ids_is_still_worth_reporting() {
        assert!(mis_addressed(1, 0).worth_reporting());
        assert!(mis_addressed(0, 1).worth_reporting());
        assert!(!mis_addressed(0, 0).worth_reporting());
        // It is a report, not a refusal -- every region carries a translation.
        assert!(!mis_addressed(3, 2).nothing_translated());

        let mut headers = HeaderMap::new();
        untranslated_headers(&mut headers, &mis_addressed(3, 2), 0, 12);
        assert_eq!(headers["x-birelate-dropped-ids"], "3/2");
        // And the pair that would otherwise have been the whole story.
        assert_eq!(headers["x-birelate-untranslated"], "0/12");
        assert_eq!(headers["x-birelate-truncated"], "false");
        // Nothing was erased and left blank, so nothing is said about that.
        assert!(headers.get("x-birelate-dropped").is_none());
    }

    /// The PNG path's half of the drop report.
    ///
    /// The `Misses::default()` here is the *measured* shape, not a convenience:
    /// on all 59 instances the translation stage emitted no
    /// `Progress::Untranslated` at all, so `segments` is 0. That is why the
    /// denominator comes from the region walk -- reusing `misses.segments` would
    /// print `4/0` -- and why the caller has to OR this into its own gate:
    /// `Misses::default().worth_reporting()` is false, so a page carrying
    /// nothing but drops would send no headers whatever.
    #[test]
    fn an_erased_and_unlettered_box_reaches_the_png_path_too() {
        assert!(!Misses::default().worth_reporting());

        let mut headers = HeaderMap::new();
        untranslated_headers(&mut headers, &Misses::default(), 4, 7);
        assert_eq!(headers["x-birelate-dropped"], "4/7");
        // A different question with a nearly identical name: that one counts
        // reply ENTRIES the translator threw away, this one counts BOXES on the
        // page that were erased and painted with nothing.
        assert!(headers.get("x-birelate-dropped-ids").is_none());
    }

    /* -------------------------------------------------------------------
     * The wiring, end to end on the CPU side.
     *
     * `regions::dropped_indices` is well covered on its own, and every one of
     * those tests stayed green while a mutation check replaced the value handed to the
     * response with `Vec::new()`. Well-tested arithmetic wired to nothing is
     * exactly the defect this whole report exists to catch, reproduced inside
     * the report -- so the three tests below stand on the three hops between the
     * finished scene and the wire, and each goes red on its own.
     * ------------------------------------------------------------------- */

    /// One region read and lettered with nothing, one ordinary. Fails if
    /// `Outcome::dropped` stops reading the regions it reports on.
    #[test]
    fn an_outcome_reports_its_own_drops() {
        let outcome = outcome_with(vec![
            dropped_region("こんにちは", "Hello"),
            // Shaped like a measured 465x198 box: read
            // correctly, answered with an empty string, erased and painted
            // with nothing.
            dropped_region("それがミナの知る『朝食』である", ""),
        ]);
        assert_eq!(outcome.dropped(), vec![1]);
        // And it is genuinely the only counter that fires: nothing was reported
        // untranslated, which is the measured shape of all 59 instances.
        assert!(outcome.untranslated.is_empty());
        assert!(!outcome.misses.worth_reporting());
    }

    /// The JSON hop. Fails if the body stops carrying what the run found.
    #[test]
    fn a_json_body_carries_the_drops_it_found() {
        let outcome = outcome_with(vec![
            dropped_region("こんにちは", "Hello"),
            dropped_region("それがミナの知る『朝食』である", ""),
            dropped_region("ふざけないでよ！！", " \n "),
        ]);
        let body = serde_json::to_value(TranslateJson::of(outcome, 42)).unwrap();
        assert_eq!(body["dropped"], serde_json::json!([1, 2]));
        assert_eq!(body["ms"], 42);
        // Beside it and not folded into it -- the two describe different
        // failures, and this is the page shape where only one of them can speak.
        assert_eq!(body["untranslated"], serde_json::json!([]));
        assert_eq!(body["truncated"], false);
    }

    /// The containment clause's wire self-description crosses through `of()`, in BOTH
    /// arms. Asserting only the false arm would stay green with the field
    /// hardcoded `false` at the crossing -- the assignable-field failure the
    /// `of()` doc records for `dropped` -- so the true arm is the half that
    /// makes this a test of the crossing rather than of a constant.
    #[test]
    fn a_json_body_says_whether_the_containment_clause_ran() {
        let off = serde_json::to_value(TranslateJson::of(outcome_with(Vec::new()), 1)).unwrap();
        assert_eq!(off["containment_clause"], false);
        let mut outcome = outcome_with(Vec::new());
        outcome.containment_clause = true;
        let on = serde_json::to_value(TranslateJson::of(outcome, 1)).unwrap();
        assert_eq!(on["containment_clause"], true);
    }

    /// The split-fragment marker crosses to the wire through `of()`, where a test can
    /// stand -- the same guard `dropped` needed after a `Vec::new()` initialiser
    /// shipped an empty list forever with the suite green.
    #[test]
    fn a_json_body_carries_the_split_fragments_it_found() {
        let outcome = outcome_with(vec![
            dropped_region("ミナにとって", "was a "),
            dropped_region("駄目", "No way."),
            // All-whitespace belongs to `dropped`, and only to `dropped`.
            dropped_region("ふざけないでよ！！", " \n "),
        ]);
        let body = serde_json::to_value(TranslateJson::of(outcome, 7)).unwrap();
        assert_eq!(body["split_fragments"], serde_json::json!([0]));
        assert_eq!(body["dropped"], serde_json::json!([2]));
    }

    /// The edge-hint hop, in both of its shapes.
    ///
    /// **The empty case is half the contract and is asserted first.** A caller
    /// that sees `edge_hints: []` must be able to read it as "the server checked
    /// this page and found none"; if the key could go missing, a joiner could
    /// not tell that from an older server and would have to guess. It is stated
    /// here rather than only in the doc comment because it is the field's one
    /// defence against being silently detached from the run -- the exact failure
    /// `dropped` had, where a `Vec::new()` initialiser left every response
    /// reporting an empty list forever with the whole suite green.
    ///
    /// The populated case carries a measured sub-floor box at its measured score, and
    /// pins that a hint is NOT a region: `regions` stays empty beside it.
    #[test]
    fn a_json_body_carries_its_edge_hints_and_an_empty_array_when_it_has_none() {
        let body = serde_json::to_value(TranslateJson::of(outcome_with(Vec::new()), 0)).unwrap();
        assert_eq!(
            body["edge_hints"],
            serde_json::json!([]),
            "an empty array means the server checked, never a missing key"
        );

        let mut outcome = outcome_with(Vec::new());
        outcome.edge_hints = vec![EdgeHintOut::from(koharu_pipeline::EdgeHint {
            x: 52.1,
            y: 712.9,
            width: 191.7,
            height: 191.6,
            edge: koharu_pipeline::PageEdge::Bottom,
            score: 0.2363,
            spans: false,
        })];
        let body = serde_json::to_value(TranslateJson::of(outcome, 0)).unwrap();
        // Widened from the f32 the pipeline reports, so the expected value has
        // to be widened the same way rather than written as a decimal literal:
        // `52.1f64` is not `f64::from(52.1f32)` and the assertion would fail on
        // the last bits of the mantissa while the field was perfectly correct.
        assert_eq!(
            body["edge_hints"],
            serde_json::json!([{
                "x": f64::from(52.1f32),
                "y": f64::from(712.9f32),
                "width": f64::from(191.7f32),
                "height": f64::from(191.6f32),
                "edge": "bottom",
                // The score rides the same widening as the geometry, and is pinned
                // BELOW the 0.25 text floor on purpose: a hint that ever serialised
                // an above-floor score would mean a region had leaked into this
                // channel, which is the one thing `mask_includes` must never see.
                "score": f64::from(0.2363f32),
                "spans": false,
            }])
        );
        assert!(
            body["edge_hints"][0]["score"].as_f64().unwrap() < 0.25,
            "a hint's score must be under the text floor -- above it, this is a region"
        );
        // A hint is not a region, and carries no index into one.
        assert_eq!(body["regions"], serde_json::json!([]));
    }

    /// The PNG hop, gate included. `Misses::default()` is the *measured* shape:
    /// on all 59 instances the translation stage emitted no
    /// `Progress::Untranslated` at all, so `worth_reporting()` is false and a
    /// gate that only consulted it would send no headers whatever.
    #[test]
    fn a_png_response_carries_the_drops_it_found() {
        let outcome = outcome_with(vec![
            dropped_region("こんにちは", "Hello"),
            dropped_region("それがミナの知る『朝食』である", ""),
        ]);
        assert!(!outcome.misses.worth_reporting());

        let mut headers = HeaderMap::new();
        miss_headers(&mut headers, &outcome);
        assert_eq!(headers["x-birelate-dropped"], "1/2");

        // And an ordinary page still answers header for header as it always has.
        let mut clean = HeaderMap::new();
        miss_headers(
            &mut clean,
            &outcome_with(vec![dropped_region("こんにちは", "Hello")]),
        );
        assert!(clean.is_empty(), "{clean:?}");
    }

    fn dropped_region(source: &str, translated: &str) -> RegionOut {
        RegionOut {
            content: koharu_scene::EntityId::new(),
            layer: koharu_scene::EntityId::new(),
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            source: source.to_owned(),
            translated: translated.to_owned(),
            fit_x: None,
            fit_y: None,
            fit_width: None,
            fit_height: None,
            source_language: None,
            target_language: None,
            detection_confidence: None,
            ocr_confidence: None,
            direction: None,
            writing_mode: None,
            region_kind: None,
            label: None,
            role: None,
            font_size: None,
            color: None,
            stroke_color: None,
            stroke_width: None,
            font_weight: None,
            refused: None,
            occluded_by: None,
        }
    }

    /// A finished run carrying nothing but its regions -- which is the measured
    /// shape of a dropped page: every other channel reads clean.
    fn outcome_with(regions: Vec<RegionOut>) -> Outcome {
        Outcome {
            png: Vec::new(),
            regions,
            untranslated: Vec::new(),
            misses: Misses::default(),
            skipped: Vec::new(),
            stages: Vec::new(),
            selected: Vec::new(),
            warnings: Vec::new(),
            rendered_text: Vec::new(),
            story_pairs: None,
            glossary_terms: None,
            containment_clause: false,
            source_script: crate::labels::SourceScript::Unknown,
            edge_hints: Vec::new(),
            placement_overflow: Vec::new(),
            placement_clamped: Vec::new(),
            translation_pins_applied: 0,
        }
    }

    /// The declared script reaches the wire beside the region stamp, and the
    /// two agree in LANGUAGE while differing in GRANULARITY.
    ///
    /// **Rewritten when the stamp began echoing the declared language, and
    /// deliberately so.** The old
    /// test pinned the DISAGREEMENT — `regions[].source_language` was `ocr.rs`'s
    /// hardcoded `ja-JP` on every region, so a zh run's two fields disagreed by
    /// falsehood. After the fix they would have kept disagreeing as *strings*
    /// (`"zh"` vs `"zh-CN"`), so the old `assert_ne!` would have stayed green
    /// describing a defect that no longer exists. The contract now: this field
    /// carries the bare script, the region field carries the declared language's
    /// full BCP-47 tag (the fixture sets what `stamped_language` stamps — the
    /// pipeline's own tests pin that function), and `ja-JP` survives only as the
    /// undeclared default. A string compare between the two fields still fails,
    /// which is why neither may be read as the other.
    #[test]
    fn the_body_reports_the_declared_script_and_the_stamp_agrees_in_language() {
        for (script, tag, region_tag) in [
            (crate::labels::SourceScript::Chinese, "zh", "zh-CN"),
            (crate::labels::SourceScript::Korean, "ko", "ko-KR"),
            (crate::labels::SourceScript::Japanese, "ja", "ja-JP"),
            // Undeclared: the stamp cannot know, so the historical default.
            (crate::labels::SourceScript::Unknown, "unknown", "ja-JP"),
        ] {
            let mut outcome = outcome_with(vec![dropped_region("嘿", "Hey")]);
            outcome.source_script = script;
            outcome.regions[0].source_language = Some(region_tag.to_owned());
            let body = TranslateJson::of(outcome, 0);
            assert_eq!(body.source_script, tag);
            assert_eq!(body.regions[0].source_language.as_deref(), Some(region_tag));
            // The granularity gap is load-bearing: equal strings would tempt a
            // reader to substitute one field for the other.
            if script != crate::labels::SourceScript::Unknown {
                assert_ne!(body.source_script, body.regions[0].source_language.as_deref().unwrap());
            }
        }
    }

    #[test]
    fn an_unknown_reading_and_a_stopped_countdown_are_null_not_zero() {
        let status = StatusJson {
            models_loaded: false,
            unload_in_secs: None,
            idle_unload_secs: 0,
            vram: vram_json(&Vram::Unknown),
            sufficient: true,
            needed_bytes: 0,
            provider: models::PROVIDER_LOCAL,
            model: models::DEFAULT_LOCAL_MODEL.to_owned(),
            busy: false,
            ocr_substitute: None,
        };
        let body = serde_json::to_value(&status).unwrap();
        assert!(body["vram"].is_null(), "{body}");
        assert!(body["unload_in_secs"].is_null(), "{body}");
        // Still present, so the popup can distinguish "no telemetry" from "the
        // server did not answer that question".
        assert!(body.get("vram").is_some());
        assert!(body.get("unload_in_secs").is_some());
    }

    #[test]
    fn the_insufficient_message_names_the_gap_inside_the_window() {
        let short = vram::Headroom {
            vram: Vram::Known {
                device: "NVIDIA GeForce RTX 5090".to_owned(),
                budget_bytes: 32 * vram::GIB,
                available_bytes: 6 * vram::GIB,
            },
            needed_bytes: 27 * vram::GIB,
            sufficient: false,
        };
        let message = insufficient_vram(&short).message;
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
        assert!(message.starts_with("insufficient VRAM: 21.0 GB short"), "{message}");

        // A hostile or broken reading must not push the actionable part out of
        // the window.
        let absurd = vram::Headroom {
            vram: Vram::Known {
                device: "x".repeat(4096),
                budget_bytes: u64::MAX,
                available_bytes: 0,
            },
            needed_bytes: u64::MAX,
            sufficient: false,
        };
        let message = insufficient_vram(&absurd).message;
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
    }

    #[test]
    fn the_reload_message_says_what_the_reader_just_did() {
        // What `vram::after_reload` produces for a shrunken budget: the whole
        // budget is the availability, because the reload hands the server's own weights
        // back before the page runs.
        let short = vram::Headroom {
            vram: Vram::Known {
                device: "NVIDIA GeForce RTX 5090".to_owned(),
                budget_bytes: 12 * vram::GIB,
                available_bytes: 12 * vram::GIB,
            },
            needed_bytes: 14 * vram::GIB,
            sufficient: false,
        };
        let message = insufficient_vram_for_reload(&short).message;
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
        // Both halves survive the window: the cause, and the gap.
        assert!(message.contains("changing the models"), "{message}");
        assert!(message.contains("2.0 GB short"), "{message}");

        let absurd = vram::Headroom {
            vram: Vram::Known {
                device: "x".repeat(4096),
                budget_bytes: 0,
                available_bytes: 0,
            },
            needed_bytes: u64::MAX,
            sufficient: false,
        };
        let message = insufficient_vram_for_reload(&absurd).message;
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
    }

    /// The real failure's own text, so the assertions below are about the message
    /// the user would actually have seen.
    const CUDNN_PANIC: &str = "called `Result::unwrap()` on an `Err` value: \
                               Torch(\"CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed \
                               ptrDesc->finalize() cudnn_status: \
                               CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH\")";

    /// Injection stands in for the cuDNN fault, which needs the GPU. What is
    /// under test is the boundary, not what crossed it.
    async fn panicking_translate() -> String {
        panic!("{CUDNN_PANIC}")
    }

    async fn body_of(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("the error body is small");
        String::from_utf8(bytes.to_vec()).expect("plain text is utf-8")
    }

    #[tokio::test]
    async fn a_panicking_handler_answers_500_rather_than_dropping_the_connection() {
        let app = guarded_with(post(panicking_translate));
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/translate")
                    .header(header::HOST, HOST)
                    .header("x-koharu-token", "s3cret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/plain; charset=utf-8")
        );
        let body = body_of(response).await;
        assert!(body.len() <= BUDGET, "{} bytes: {body}", body.len());
        // The point of the whole message path: the scaffolding is gone and the
        // word the user has to act on is inside the window.
        assert!(body.starts_with("the server panicked: CUDNN_BACKEND"), "{body}");
        assert!(body.contains("cudnnFinalize"), "{body}");

        // Still serving, on the same router the panic went through.
        for (method, uri) in [("GET", "/health"), ("GET", "/status")] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header(header::HOST, HOST)
                        .header("x-koharu-token", "s3cret")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{method} {uri}");
        }
    }

    #[tokio::test]
    async fn a_panicking_device_section_frees_the_permit_and_restarts_the_countdown() {
        // `translate_page`'s shape with the device section replaced by a panic:
        // building the real one needs a `Pipeline`, and what is at stake here is
        // the frame, not what runs inside it. A leaked permit would wedge every
        // later request, and an unarmed countdown would leave the weights
        // resident for the rest of the process.
        let gate = Arc::new(tokio::sync::Mutex::new(()));
        let clock = crate::idle::IdleClock::new(Some(Duration::from_secs(300)));

        let outcome: Result<(), ApiError> = {
            let _idle = TouchOnDrop(&clock);
            let _gpu = gate.clone().lock_owned().await;
            AssertUnwindSafe(async {
                // Yielding first makes the panic land on a later poll, which is
                // where a real stage failure happens.
                tokio::task::yield_now().await;
                panic!("{CUDNN_PANIC}")
            })
            .catch_unwind()
            .await
            .unwrap_or_else(|payload| Err(panicked("the translation", &*payload)))
        };

        let error = outcome.expect_err("the device section panicked");
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            error.message.len() <= BUDGET,
            "{} bytes: {}",
            error.message.len(),
            error.message
        );
        assert!(error.message.contains("cudnnFinalize"), "{}", error.message);
        assert!(gate.try_lock().is_ok(), "the GPU permit leaked");
        assert!(
            clock.remaining().is_some(),
            "the idle deadline was left unarmed"
        );
    }

    #[tokio::test]
    async fn a_panicked_task_is_reported_by_its_message_and_a_cancelled_one_is_not() {
        let panicked_task = tokio::spawn(async { panic!("{CUDNN_PANIC}") })
            .await
            .expect_err("the task panicked");
        let error = task_failed("the translation", panicked_task);
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(error.message.len() <= BUDGET, "{}", error.message);
        assert!(error.message.contains("cudnnFinalize"), "{}", error.message);

        // A cancelled task carries no payload at all, so `into_panic` would
        // itself panic; the caller gets the generic line instead.
        let handle = tokio::spawn(async { std::future::pending::<()>().await });
        handle.abort();
        let cancelled = handle.await.expect_err("the task was cancelled");
        assert_eq!(
            task_failed("the translation", cancelled).message,
            RENDER_TASK_FAILED
        );
    }

    #[tokio::test]
    async fn no_response_ever_carries_a_permissive_cors_header() {
        let response = guarded()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header(header::HOST, HOST)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            response
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .is_none()
        );
    }
}
