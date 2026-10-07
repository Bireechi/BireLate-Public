//! A local HTTP wrapper around koharu's headless pipeline, shaped for the
//! BireLate Firefox extension.
//!
//! One process, one pipeline, one page per request. Nothing is written to disk:
//! both configurations are in-memory handles, so the server can never clobber
//! the settings the koharu desktop app keeps in the user's home directory.

pub mod cli;
pub mod downloads;
pub mod duplicate;
pub mod engine;
pub mod error;
pub mod form;
pub mod glossary;
pub mod guard;
pub mod idle;
pub mod labels;
pub mod lettering;
pub mod models;
pub mod pins;
pub mod placement;
pub mod regions;
pub mod render;
pub mod routes;
pub mod sfx;
pub mod scene;
pub mod story;
pub mod vram;
pub mod watch;

use std::{
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

use anyhow::{Context as _, Result};
use koharu_config::Config;
use koharu_pipeline::Pipeline;
use tracing_subscriber::EnvFilter;

pub use cli::Cli;

use crate::{
    cli::Resolved,
    engine::{AppState, GpuState, Selection, Shared, desired_config},
    guard::GuardState,
    idle::IdleClock,
    render::Renderers,
};

/// How long startup waits for the resource monitor's first publication.
///
/// Only to stop the first `/status` reporting VRAM as unknown; the wait is
/// allowed to fail. koharu's own internal wait uses the same two seconds.
const FIRST_VRAM_SAMPLE: Duration = Duration::from_secs(2);

pub async fn run(cli: Cli) -> Result<()> {
    let (filter, filter_source) = log_filter();
    // Rendered before the filter is moved into the subscriber. `EnvFilter`
    // displays the directives it actually parsed, so this reports what took
    // effect rather than echoing back what was typed.
    let filter_directives = filter.to_string();
    tracing_subscriber::fmt().with_env_filter(filter).init();
    log_panics();
    // Announced because the last thing to go wrong here went wrong in silence:
    // a filter the server never read produced a log with none of the requested
    // lines in it, and an empty grep reads as "that code did not run" rather
    // than "you asked the wrong process". One line at startup settles it.
    tracing::info!(
        source = filter_source,
        directives = %filter_directives,
        "log filter in effect"
    );

    // Before anything can read `Store::root()`: the first read fixes the root
    // for the life of the process, and a later `configure` is then an error.
    if let Some(dir) = cli.resolve_store_dir(std::env::var_os("BIRELATE_STORE_DIR"))? {
        koharu_runtime::Store::configure(dir)?;
    }
    let resolved = cli.resolve(std::env::var("BIRELATE_TOKEN").ok())?;
    announce_startup(&resolved);
    if let Some(engine) = &resolved.defaults.ocr_substitute {
        tracing::info!(
            "--hunyuan-substitute: requests for hunyuan-ocr-1.5 are served by {}",
            models::ocr_model_name(engine)
        );
    }

    // Subscribed before `init`, the first thing that downloads: a receiver
    // only sees events sent after it exists.
    downloads::log_progress();
    initialize_with_retry().await;
    let device = koharu_ml::device(resolved.cpu);

    let initial = desired_config(
        &Selection {
            ocr: resolved.defaults.ocr.clone(),
            inpainting: resolved.defaults.inpainting.clone(),
            target_language: resolved.defaults.target_language,
            // Nothing has declared a source language at startup -- there is no
            // request yet -- and `Unknown` is the state in which no script rule
            // fires. A request that declares one reloads the config, which is
            // affordable only because the extension latches it PER HOST.
            source_script: labels::SourceScript::Unknown,
            llm: resolved.pinned.llm.clone(),
            // The startup config takes the process default; a request that
            // carries the popup toggle reloads, exactly as one carrying a
            // different OCR engine does.
            segment_context: resolved.defaults.segment_context,
        },
        &resolved.defaults,
        &resolved.pinned,
    );

    // Exactly one pipeline, for the life of the process. Its config watcher
    // owns the channel that keeps itself alive, so it can never exit; dropping
    // the pipeline would strand the loaded weights instead of freeing them.
    let pipeline = Pipeline::from_config(
        Config::memory(initial.clone()),
        Config::memory(resolved.providers),
        device,
    )
    .context("failed to build the translation pipeline")?;

    // Built here rather than lazily so a machine without a usable graphics
    // adapter fails at startup instead of on the user's first page.
    let renderers = tokio::task::spawn_blocking(Renderers::new)
        .await
        .context("the renderer setup task failed")?
        .context("failed to start the renderer; try setting WGPU_BACKEND=dx12 or WGPU_BACKEND=vulkan")?;

    tracing::info!(
        provider = resolved.pinned.wire_provider,
        llm = %resolved.pinned.llm,
        "translation backend is fixed for this process"
    );

    // Subscribed here, not lazily in a handler: the monitor only starts sampling
    // from inside a Tokio runtime, and outside one it silently leaves the channel
    // at its default for ever.
    let vram = pipeline.subscribe_resources();
    {
        // A fresh receiver starts at the version the channel is already on, so
        // this resolves on the *first* publication rather than waiting for a
        // second one.
        let mut first = vram.clone();
        if tokio::time::timeout(FIRST_VRAM_SAMPLE, first.changed())
            .await
            .is_err()
        {
            tracing::warn!("no VRAM sample yet; /status reports it as unknown until one lands");
        }
    }

    let watch_pid = resolved.watch_pid;
    let state = AppState(Arc::new(Shared {
        pipeline,
        renderers: Arc::new(renderers),
        gate: Arc::new(tokio::sync::Mutex::new(GpuState { applied: initial })),
        defaults: resolved.defaults,
        pinned: resolved.pinned,
        guard: GuardState {
            token_digest: resolved.token_digest,
            allowed_origins: resolved.allowed_origins.into(),
            allowed_hosts: resolved.allowed_hosts.into(),
        },
        font_families: resolved.font_families,
        hyphenation: resolved.hyphenation,
        size_coherence: resolved.size_coherence,
        uppercase_dialogue: resolved.uppercase_dialogue,
        skip_implausible_text: resolved.skip_implausible_text,
        scope_watermark_refusals: resolved.scope_watermark_refusals,
        leave_misread_bubbles: resolved.leave_misread_bubbles,
        korean_script_strict: resolved.korean_script_strict,
        skip_duplicate_text: resolved.skip_duplicate_text,
        duplicate_oriented_overlap: resolved.duplicate_oriented_overlap,
        duplicate_shared_source: resolved.duplicate_shared_source,
        collision_relief: resolved.collision_relief,
        edge_anchored_lettering: resolved.edge_anchored_lettering,
        fit_free_text: resolved.fit_free_text,
        source_ink_fraction: resolved.source_ink_fraction,
        sfx_dictionary: resolved.sfx_dictionary,
        max_upload_bytes: resolved.max_upload_bytes,
        queue_timeout: resolved.queue_timeout,
        vram,
        idle: IdleClock::new(resolved.idle_unload),
        warming: AtomicBool::new(false),
        cold_reserve_bytes: resolved.cold_reserve_bytes,
        shutdown: std::sync::Arc::new(tokio::sync::Notify::new()),
        stories: story::Stories::new(resolved.story_pairs),
        story_excludes_sfx: resolved.story_excludes_sfx,
        glossaries: glossary::Glossaries::new(),
    }));

    tokio::spawn(idle::watch(state.clone()));
    if let Some(pid) = watch_pid {
        // The no-orphans tie: when the launcher dies, so do we --
        // through the same graceful path as POST /shutdown, so an in-flight
        // page still finishes. The Hunyuan shim carries the same tie in Python.
        // A std thread, NOT a tokio blocking task -- watch.rs says why.
        watch::spawn_watch(pid, state.shutdown.clone());
    }
    if resolved.warmup {
        // Measured even though the flag is explicit. On a card another process
        // has filled, loading these models is a native abort with exit code 3,
        // no Rust panic and no output -- and here that would happen at startup,
        // where the user has nothing to read but a vanished process.
        let headroom = vram::headroom(&state.vram, false, state.cold_reserve_bytes);
        if !headroom.sufficient {
            tracing::warn!(
                needed_bytes = headroom.needed_bytes,
                "--warmup skipped: not enough VRAM for a cold start right now"
            );
        } else if state.claim_warmup() {
            tracing::info!("--warmup: loading the models before the first page");
            tokio::spawn(engine::warm_models(state.clone()));
        }
    }

    /* Before the first page, not after: these hold the reader's manga, and the
     * point is that a crashed earlier run does not leave it lying in %TEMP%. */
    let swept = scene::sweep_stale_storage();
    if swept > 0 {
        tracing::info!(swept, "cleared page data left by an earlier run");
    }

    let listener = tokio::net::TcpListener::bind(resolved.addr)
        .await
        .with_context(|| format!("failed to bind {}", resolved.addr))?;
    tracing::info!(addr = %resolved.addr, "birelate-server listening");

    /* Either signal ends the process, and both go through axum's graceful path
     * so an in-flight page still finishes and still gets its response. The
     * popup's Stop button is the one the reader actually uses; ctrl_c is for a
     * server started from a console. */
    let shutdown = state.shutdown.clone();
    axum::serve(listener, routes::app(state))
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => tracing::info!("ctrl-c: shutting down"),
                () = shutdown.notified() => tracing::info!("stop requested: shutting down"),
            }
        })
        .await
        .context("the server stopped unexpectedly")?;
    Ok(())
}

/// The default directives, used when neither environment variable is set.
///
/// The leading bare `info` is load-bearing and not a tidier way of writing the
/// two that follow: it is the default level for *every* target, which is what
/// puts koharu's own `info` events -- notably the residency evictions in
/// `koharu_pipeline::residency`, the single most expensive decision a run takes
/// -- into an unconfigured log. Narrowing it to `birelate_server=info` would
/// silence the pipeline this process exists to drive.
const DEFAULT_LOG_FILTER: &str = "info,birelate_server=info,tower_http=info";

/// Picks the log directives and names where they came from: `BIRELATE_LOG`,
/// then `RUST_LOG`, then the default above.
///
/// `RUST_LOG` is in the chain because leaving it out cost a whole measurement
/// run. `EnvFilter::try_from_env("BIRELATE_LOG")` reads that name and *only*
/// that name, so a chapter launched with
/// `RUST_LOG="info,koharu_pipeline::residency=debug"` silently got the default
/// filter, and grepping the resulting log for `residency` returned nothing --
/// which is indistinguishable from the code path never having run. Rust
/// binaries conventionally honour `RUST_LOG`, so being one that does not is a
/// trap rather than a policy.
///
/// `BIRELATE_LOG` still wins where both are set: this process shares a shell
/// with cargo and with koharu's own binaries, and a `RUST_LOG` aimed at one of
/// those should not quietly retune the server.
///
/// Empty and unparseable values fall through to the next source instead of
/// taking effect. An empty `EnvFilter` parses cleanly and enables **nothing**,
/// so `$env:BIRELATE_LOG = ""` -- the obvious way to undo a filter in
/// PowerShell, since the variable still exists afterwards -- would otherwise
/// produce a completely silent server.
///
/// Split out from `log_filter` so the precedence can be tested: reading the
/// environment inside a test races every other test in the binary.
fn log_directives(birelate: Option<&str>, rust: Option<&str>) -> (String, &'static str) {
    for (value, source) in [(birelate, "BIRELATE_LOG"), (rust, "RUST_LOG")] {
        let Some(directives) = value.map(str::trim).filter(|d| !d.is_empty()) else {
            continue;
        };
        if EnvFilter::try_new(directives).is_ok() {
            return (directives.to_owned(), source);
        }
    }
    (DEFAULT_LOG_FILTER.to_owned(), "default")
}

fn log_filter() -> (EnvFilter, &'static str) {
    let birelate = std::env::var("BIRELATE_LOG").ok();
    let rust = std::env::var("RUST_LOG").ok();
    let (directives, source) = log_directives(birelate.as_deref(), rust.as_deref());
    (EnvFilter::new(directives), source)
}

/// Sends every panic through the log as well as stderr.
///
/// A panic on a Tokio worker or in the blocking pool is delivered to whoever
/// awaits that task and to nobody else, so the tasks spawned here and never
/// joined -- the idle watcher, the warmup -- lose theirs entirely: nothing
/// correlates them with the request that provoked them, and nothing puts them
/// beside the other lines the operator is reading. Chained rather than replaced,
/// so the default report keeps going to stderr exactly as before.
///
/// Note what this cannot see. A native `abort()` -- exit code 3 with no output --
/// is not a Rust panic, runs no hook, and leaves nothing here to log. Silence
/// from this hook next to a dead process is therefore itself the diagnosis: the
/// process was killed below Rust.
fn log_panics() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        tracing::error!(
            thread = thread.name().unwrap_or("<unnamed>"),
            location = %info.location().map_or_else(
                || "unknown".to_owned(),
                std::string::ToString::to_string,
            ),
            panic = %error::panic_text(info.payload()),
            "a thread panicked"
        );
        previous(info);
    }));
}

/// Printed to stdout rather than logged, because the user has to act on it.
fn announce_startup(resolved: &Resolved) {
    println!("+----------------------------------------------------------+");
    // The extension's own default is "ollama" while this server's is "local",
    // so a fresh install 409s on its first translate unless one of them moves.
    // Naming the value here is cheaper than making the user read the 409.
    println!(
        "  Set the popup's Provider to '{}' and its Model to",
        resolved.pinned.wire_provider
    );
    println!("  '{}', or leave the Model box blank.", resolved.pinned.llm);
    match &resolved.generated_token {
        Some(token) => {
            println!("  Token: {token}");
            println!("  Paste it into the popup. Set --token or BIRELATE_TOKEN");
            println!("  to keep the same one across restarts.");
        }
        None if resolved.token_digest.is_none() => {
            println!("  WARNING: running with --no-token. Only the Origin and");
            println!("  Host guards stand between this server and any web page");
            println!("  you visit. Prefer a token.");
        }
        None => {}
    }
    // Configured or default, so the reader knows where the gigabytes go.
    println!("  Model store: {}", koharu_runtime::Store::root().display());
    println!("  (move it with --store-dir or BIRELATE_STORE_DIR)");
    println!("+----------------------------------------------------------+");
}

/// Copied from the headless CLI: the native runtimes are downloaded on first
/// use, so a cold start can legitimately fail a few times before succeeding.
async fn initialize_with_retry() {
    let mut delay = Duration::from_secs(1);
    let mut attempt = 0_u64;
    loop {
        attempt += 1;
        match koharu_ml::init().await {
            Ok(()) => return,
            Err(error) => {
                let jitter = Duration::from_millis((attempt.wrapping_mul(137)) % 251);
                let wait = delay + jitter;
                tracing::warn!(
                    attempt,
                    %error,
                    "runtime initialization failed; retrying in {:.1}s",
                    wait.as_secs_f64()
                );
                tokio::time::sleep(wait).await;
                delay = delay.saturating_mul(2).min(Duration::from_secs(30));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn birelate_log_wins_over_rust_log() {
        assert_eq!(
            log_directives(Some("warn"), Some("trace")),
            ("warn".to_owned(), "BIRELATE_LOG")
        );
    }

    #[test]
    fn rust_log_is_honoured_when_birelate_log_is_unset() {
        // The whole point of the change: this exact string was set on a chapter
        // run, was read by nothing, and the resulting log had zero lines
        // mentioning residency.
        assert_eq!(
            log_directives(None, Some("info,koharu_pipeline::residency=debug")),
            (
                "info,koharu_pipeline::residency=debug".to_owned(),
                "RUST_LOG"
            )
        );
    }

    #[test]
    fn an_empty_variable_falls_through_rather_than_silencing_the_server() {
        assert_eq!(
            log_directives(Some("   "), Some("debug")),
            ("debug".to_owned(), "RUST_LOG")
        );
        assert_eq!(
            log_directives(Some(""), None),
            (DEFAULT_LOG_FILTER.to_owned(), "default")
        );
    }

    #[test]
    fn an_unparseable_variable_falls_through_rather_than_taking_effect() {
        assert_eq!(
            log_directives(Some("=========="), None),
            (DEFAULT_LOG_FILTER.to_owned(), "default")
        );
    }

    #[test]
    fn the_default_filter_lets_koharus_own_info_events_through() {
        // A bare `info` directive ahead of the per-crate ones is what carries
        // koharu_pipeline's residency evictions into an unconfigured log. Losing
        // it would make the eviction invisible again without changing a line of
        // koharu.
        let (directives, source) = log_directives(None, None);
        assert_eq!(source, "default");
        assert!(
            directives.split(',').any(|directive| directive == "info"),
            "the default filter must keep a bare global `info`: {directives}"
        );
        assert!(EnvFilter::try_new(&directives).is_ok());
    }
}
