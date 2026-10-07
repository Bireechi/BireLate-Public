//! The idle budget: models are freed when nothing has used them for a while.
//!
//! The countdown starts when a translation *finishes*, not when it starts. A
//! page can take minutes, and a deadline measured from the start would fire
//! while the run it was meant to outlast is still going.
//!
//! Freeing has to drop the weights without replacing the stage runner. koharu
//! keeps the learned per-stage VRAM profiles inside the runner, and an unprofiled
//! stage is admitted by evicting every other model -- including itself -- so it
//! can be measured alone. Rebuilding the runner to free memory would therefore
//! make the first page after every nap as slow as a cold start, permanently.
//! That is why freeing goes through `Pipeline::unload_models`, which drops the
//! weights and keeps the runner, and why `Pipeline::reload` is not used here.
//!
//! The clock's arithmetic takes `now` as an argument so it can be tested at
//! whatever instant the test likes, without sleeping.

use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

use crate::engine::{AppState, free_models};

/// When the countdown started, if it is running at all. Pure: every method that
/// needs the time is given it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Countdown {
    armed: Option<Instant>,
}

impl Countdown {
    pub fn arm(&mut self, now: Instant) {
        self.armed = Some(now);
    }

    pub fn disarm(&mut self) {
        self.armed = None;
    }

    /// How long is left, or `None` when the countdown is not running.
    ///
    /// `Instant + Duration` panics on overflow and the budget arrives as a `u64`
    /// of seconds from the command line, so the addition is checked: a budget
    /// that cannot be represented never expires, which is the right reading of
    /// an absurdly large one.
    #[must_use]
    pub fn remaining(self, budget: Duration, now: Instant) -> Option<Duration> {
        let armed = self.armed?;
        Some(match armed.checked_add(budget) {
            Some(deadline) => deadline.saturating_duration_since(now),
            None => Duration::MAX,
        })
    }

    #[must_use]
    pub fn expired(self, budget: Duration, now: Instant) -> bool {
        self.remaining(budget, now) == Some(Duration::ZERO)
    }
}

/// The countdown, its budget, and the way to interrupt a sleep on it.
pub struct IdleClock {
    /// `None` disables the whole feature: `--idle-unload-secs 0`.
    budget: Option<Duration>,
    countdown: Mutex<Countdown>,
    /// Woken when a translation finishes, so the watcher re-reads the deadline
    /// instead of freeing models on the previous one.
    wake: tokio::sync::Notify,
}

impl IdleClock {
    #[must_use]
    pub fn new(budget: Option<Duration>) -> Self {
        Self {
            budget,
            countdown: Mutex::new(Countdown::default()),
            wake: tokio::sync::Notify::new(),
        }
    }

    #[must_use]
    pub fn budget(&self) -> Option<Duration> {
        self.budget
    }

    /// Something finished that may have left weights resident. Restarts the
    /// countdown from now.
    pub fn touch(&self) {
        if self.budget.is_none() {
            return;
        }
        self.locked().arm(Instant::now());
        self.wake.notify_one();
    }

    /// The weights are gone; there is nothing left to free.
    pub fn cleared(&self) {
        self.locked().disarm();
    }

    /// What `/status` reports as `unload_in_secs`. `None` whenever no unload is
    /// pending, which covers a disabled budget and a server holding nothing.
    #[must_use]
    pub fn remaining(&self) -> Option<Duration> {
        let budget = self.budget?;
        self.locked().remaining(budget, Instant::now())
    }

    /// A poisoned lock here means a panic while holding a `Countdown`, which
    /// carries one `Option<Instant>` and cannot be left half-written.
    fn locked(&self) -> std::sync::MutexGuard<'_, Countdown> {
        self.countdown
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

/// Restarts the countdown however the run that owns it ends -- a `?`, a panic
/// caught above it, or success.
///
/// It borrows the clock rather than holding an `AppState` so that this, the one
/// piece of the translate path that has to survive a stage dying, can be tested
/// without a `Pipeline`.
pub struct TouchOnDrop<'a>(pub &'a IdleClock);

impl Drop for TouchOnDrop<'_> {
    fn drop(&mut self) {
        self.0.touch();
    }
}

/// Frees the models once the budget runs out, for as long as the process lives.
///
/// Spawned at startup and never joined. It returns immediately when the budget
/// is disabled, so the task is not kept alive for nothing.
pub async fn watch(state: AppState) {
    let Some(budget) = state.idle.budget() else {
        tracing::info!("idle unload disabled");
        return;
    };
    tracing::info!(secs = budget.as_secs(), "models are freed after this long idle");
    loop {
        match state.idle.remaining() {
            // Nothing is loaded, so there is nothing to count down to. Sleeping
            // on the notification costs nothing until a page arrives.
            None => state.idle.wake.notified().await,
            Some(left) if left.is_zero() => free_when_still_idle(&state, budget).await,
            Some(left) => {
                tokio::select! {
                    () = tokio::time::sleep(left) => {}
                    () = state.idle.wake.notified() => {}
                }
            }
        }
    }
}

/// Takes the GPU permit before freeing anything.
///
/// This is what makes the unload correct rather than merely likely: koharu's
/// unload gives up on a model cell that is locked instead of waiting for it, and
/// reports the same `false` it reports for a cell that was already empty. Holding
/// the gate means no cell can be locked.
///
/// Waiting for the gate can take as long as a page does, so the deadline is
/// re-read afterwards: a translation that finished while we queued has already
/// restarted the countdown, and freeing its weights immediately would be the
/// exact opposite of what the budget is for.
async fn free_when_still_idle(state: &AppState, budget: Duration) {
    let _gpu = state.gate.lock().await;
    if !state.idle.locked().expired(budget, Instant::now()) {
        return;
    }
    let freed = free_models(&state.pipeline);
    state.idle.cleared();
    if freed {
        tracing::info!(secs = budget.as_secs(), "freed the models after an idle period");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUDGET: Duration = Duration::from_secs(300);

    fn at(base: Instant, secs: u64) -> Instant {
        base + Duration::from_secs(secs)
    }

    #[test]
    fn a_countdown_that_was_never_armed_never_expires() {
        let countdown = Countdown::default();
        let now = Instant::now();
        assert_eq!(countdown.remaining(BUDGET, now), None);
        assert!(!countdown.expired(BUDGET, at(now, 86_400)));
    }

    #[test]
    fn the_deadline_is_the_budget_after_the_last_finish() {
        let base = Instant::now();
        let mut countdown = Countdown::default();
        countdown.arm(base);
        assert_eq!(
            countdown.remaining(BUDGET, base),
            Some(Duration::from_secs(300))
        );
        assert_eq!(
            countdown.remaining(BUDGET, at(base, 23)),
            Some(Duration::from_secs(277))
        );
        assert!(!countdown.expired(BUDGET, at(base, 299)));
    }

    #[test]
    fn expiry_is_the_moment_the_budget_runs_out_and_stays_expired() {
        let base = Instant::now();
        let mut countdown = Countdown::default();
        countdown.arm(base);
        assert!(countdown.expired(BUDGET, at(base, 300)));
        // Saturating, not wrapping: an overslept deadline must not look fresh.
        assert!(countdown.expired(BUDGET, at(base, 86_400)));
        assert_eq!(
            countdown.remaining(BUDGET, at(base, 86_400)),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn a_second_page_pushes_the_deadline_out() {
        let base = Instant::now();
        let mut countdown = Countdown::default();
        countdown.arm(base);
        countdown.arm(at(base, 290));
        assert!(!countdown.expired(BUDGET, at(base, 400)));
        assert_eq!(
            countdown.remaining(BUDGET, at(base, 400)),
            Some(Duration::from_secs(190))
        );
        assert!(countdown.expired(BUDGET, at(base, 590)));
    }

    #[test]
    fn freeing_the_models_stops_the_countdown() {
        let base = Instant::now();
        let mut countdown = Countdown::default();
        countdown.arm(base);
        countdown.disarm();
        assert_eq!(countdown.remaining(BUDGET, at(base, 900)), None);
        assert!(!countdown.expired(BUDGET, at(base, 900)));
    }

    #[test]
    fn an_unrepresentable_budget_never_fires_instead_of_panicking() {
        // --idle-unload-secs takes a u64, so `Instant + budget` can overflow.
        let base = Instant::now();
        let mut countdown = Countdown::default();
        countdown.arm(base);
        let forever = Duration::from_secs(u64::MAX);
        assert_eq!(countdown.remaining(forever, base), Some(Duration::MAX));
        assert!(!countdown.expired(forever, at(base, 86_400)));
    }

    #[test]
    fn a_disabled_budget_reports_nothing_to_the_popup() {
        let clock = IdleClock::new(None);
        clock.touch();
        assert_eq!(clock.budget(), None);
        // Never a countdown of zero: the popup would render "unloads in 0:00"
        // for a server that will never unload.
        assert_eq!(clock.remaining(), None);
    }

    #[test]
    fn the_countdown_restarts_even_when_the_run_it_guards_panics() {
        // A stage that dies still leaves the stages before it resident, so the
        // budget has to start counting whether the run ended well or not.
        let clock = IdleClock::new(Some(BUDGET));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _idle = TouchOnDrop(&clock);
            panic!("CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed");
        }));
        assert!(outcome.is_err());
        assert!(
            clock.remaining().is_some(),
            "a panicking run left the idle deadline unarmed"
        );
    }

    #[test]
    fn a_clock_reports_nothing_until_something_has_been_loaded() {
        let clock = IdleClock::new(Some(BUDGET));
        assert_eq!(clock.remaining(), None);
        clock.touch();
        let left = clock.remaining().expect("a touched clock counts down");
        assert!(left <= BUDGET && left > BUDGET - Duration::from_secs(5), "{left:?}");
        clock.cleared();
        assert_eq!(clock.remaining(), None);
    }
}
