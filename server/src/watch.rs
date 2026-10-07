//! Shuts the server down when the process that launched it is gone.
//!
//! The no-orphans rule: a session that dies must take its whole stack with it.
//! The Hunyuan shim takes a `--watch-pid`; before the
//! server took one too, a kill test showed the gap: `serve.ps1` was killed, the
//! sidecar stack folded in 5.4 s, and **`birelate-server` kept serving** —
//! Windows children outlive their parents, and nothing watched ours. A warmed
//! orphan holds ~20 GB of VRAM that nobody will free, which is strictly worse
//! than the shim's dead port.
//!
//! The watch is one `WaitForSingleObject` on a `SYNCHRONIZE` handle, parked on
//! a detached `std` thread — no polling, no timer. When the watched process exits
//! (or cannot be opened, which means it was gone before we looked), the
//! server's own graceful-shutdown `Notify` fires: the same path as
//! `POST /shutdown`, so an in-flight page still finishes and still answers.
//!
//! PID-reuse caveat, accepted and documented rather than defended against: if
//! the launcher dies before this server opens the handle AND the OS re-issues
//! the PID in that window, the watch binds to the wrong process. The window is
//! the server's own startup (milliseconds), and the failure mode is the old
//! behaviour (an orphan), not a wrong kill.

use std::sync::Arc;

use tokio::sync::Notify;

/// Waits — blocking — until the process is gone, then reports how it knew.
///
/// Split from [`shutdown_when_gone`] so the OS half can be exercised by tests
/// that have no `Notify` and no runtime.
#[cfg(windows)]
pub fn wait_for_exit(pid: u32) -> &'static str {
    // Hand-rolled kernel32 bindings rather than a `windows-sys` dependency:
    // three functions, used once, on the only platform this project ships on.
    use std::os::raw::c_void;
    type Handle = *mut c_void;
    const SYNCHRONIZE: u32 = 0x0010_0000;
    const INFINITE: u32 = u32::MAX;
    unsafe extern "system" {
        fn OpenProcess(desired_access: u32, inherit_handle: i32, process_id: u32) -> Handle;
        fn WaitForSingleObject(handle: Handle, milliseconds: u32) -> u32;
        fn CloseHandle(handle: Handle) -> i32;
    }

    // SAFETY: OpenProcess with a PID we do not own returns null rather than
    // faulting; a null handle is never waited on or closed.
    let handle = unsafe { OpenProcess(SYNCHRONIZE, 0, pid) };
    if handle.is_null() {
        // Gone before we looked (or never ours to open, which for the
        // same-user launcher this flag is for does not happen).
        return "unopenable, treated as already gone";
    }
    // SAFETY: the handle is non-null and stays owned by this frame; INFINITE
    // parks this thread until the process object signals, which for a process
    // handle is its termination.
    unsafe {
        WaitForSingleObject(handle, INFINITE);
        CloseHandle(handle);
    }
    "exited"
}

#[cfg(not(windows))]
pub fn wait_for_exit(_pid: u32) -> &'static str {
    // The project ships on Windows only; a non-Windows build gets no watch
    // rather than a wrong one. Stated loudly so a port cannot inherit silence.
    "watch-pid is not implemented off Windows; no watch is running"
}

/// Parks a plain `std` thread on the watched process and fires the server's
/// graceful shutdown when it is gone.
///
/// A **`std::thread`, not `spawn_blocking`, and the difference is an observed
/// regression.** Tokio's runtime waits for its blocking pool on drop, and on
/// the CLEAN path — `POST /shutdown` with the launcher
/// still alive — this thread is parked in `WaitForSingleObject(INFINITE)`
/// forever. As a blocking task it wedged the whole exit: the listener closed,
/// `main` returned from `run()`, and the process then hung under the runtime's
/// drop, so `serve.ps1` never got past its foreground call and never stopped
/// the sidecars. Observed: every process alive 60 s after a 200 from
/// `/shutdown`. A detached `std` thread does not block process exit —
/// it simply dies with the process, which is exactly the semantics a watchdog
/// wants on both paths.
///
/// `notify_one` rather than `notify_waiters`, deliberately: it stores a permit
/// when nobody is waiting yet, so a launcher that dies during our own startup
/// — before `run()` reaches its `notified().await` — still stops the server
/// instead of losing the wakeup.
pub fn spawn_watch(pid: u32, shutdown: Arc<Notify>) {
    let spawned = std::thread::Builder::new()
        .name("watch-pid".into())
        .spawn(move || {
            let how = wait_for_exit(pid);
            tracing::warn!(pid, how, "watched launcher is gone; shutting down (--watch-pid)");
            shutdown.notify_one();
        });
    if let Err(error) = spawned {
        // No watch is the pre-flag behaviour; run without one rather than die,
        // but say so — silence here would read as the tie working.
        tracing::error!(%error, pid, "could not start the --watch-pid thread; running unwatched");
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use std::process::Command;
    use std::time::Duration;

    /// The fire case, through the same function `run()` calls: a real child
    /// process exits and the shutdown notify resolves.
    #[tokio::test]
    async fn a_child_that_exits_fires_the_shutdown_notify() {
        let child = Command::new("cmd")
            .args(["/c", "ver"])
            .spawn()
            .expect("cmd /c ver should spawn");
        let shutdown = Arc::new(Notify::new());
        spawn_watch(child.id(), shutdown.clone());
        tokio::time::timeout(Duration::from_secs(10), shutdown.notified())
            .await
            .expect("the notify must fire once the child exits");
    }

    /// The not-fire case: a live process does not shut the server down — and
    /// killing it then does, which is the whole mechanism end to end.
    #[tokio::test]
    async fn a_live_child_does_not_fire_until_it_is_killed() {
        let mut child = Command::new("ping")
            .args(["-n", "30", "127.0.0.1"])
            .spawn()
            .expect("ping should spawn");
        let shutdown = Arc::new(Notify::new());
        spawn_watch(child.id(), shutdown.clone());
        assert!(
            tokio::time::timeout(Duration::from_millis(1500), shutdown.notified())
                .await
                .is_err(),
            "a live watched process must not trigger a shutdown"
        );
        child.kill().expect("killing our own ping");
        tokio::time::timeout(Duration::from_secs(10), shutdown.notified())
            .await
            .expect("the notify must fire once the watched process is killed");
    }
}
