use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use crate::{
    ResourceSnapshot, Stage,
    resources::{DeviceMemoryMeasurement, ResourceMonitor, selected_device},
    stages::Stages,
};

pub(crate) struct Residency {
    resources: Arc<ResourceMonitor>,
    // Heterogeneous CUDA model pairs took 2.5-4x longer together than
    // back-to-back on the target workload. The fair lane protects throughput;
    // VRAM profiles below still decide which weights remain resident.
    lane: Arc<tokio::sync::Semaphore>,
    sequence: AtomicU64,
    state: Mutex<State>,
    // Whether the sweep asks torch to hand its cached segments back to the
    // driver. See `releases_cached_device_memory` for who decides, and
    // `ProcessorConfig::release_cached_vram` for why it is a decision at all.
    release_cached_device_memory: bool,
}

#[derive(Default)]
struct State {
    profiles: BTreeMap<Stage, ModelProfile>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ModelProfile {
    resident_bytes: u64,
    workspace_bytes: u64,
    peak_bytes: u64,
}

pub(crate) struct Admission<'a> {
    residency: Option<&'a Residency>,
    _lane: Option<tokio::sync::OwnedSemaphorePermit>,
    profiling: bool,
    was_loaded: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AdmissionPlan {
    profiling: bool,
    unload_idle: bool,
    /// Carried purely so the caller can log the arithmetic. Nothing branches on
    /// it: `admission_plan` has already decided by the time it is set.
    reason: AdmissionReason,
}

/// Which of `admission_plan`'s four exits was taken, and the numbers behind it.
///
/// An eviction is the most expensive thing this module does -- it can destroy a
/// 16 GiB translator to admit a 153 MiB detector, and the next page then pays a
/// cold reload -- yet the decision reduced to two booleans that no log line ever
/// carried. Naming the exit makes an eviction greppable; carrying the terms
/// makes it explicable without an external VRAM sampler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdmissionReason {
    /// No profile for this stage yet, so it has to run with the card to itself
    /// or the measurement is of everyone's weights at once. Evicts regardless of
    /// how much is free -- the budget is not consulted and would not change it.
    Unprofiled,
    /// The device published no budget or no availability. Evicts, because with
    /// no arithmetic possible the safe guess is that there is no room.
    NoTelemetry,
    /// Both terms were known and compared. The only exit that can decline to
    /// evict, and the only one with numbers worth printing.
    Budgeted(Budgeted),
    /// `recover` rather than `enter`: a stage already failed with an
    /// out-of-memory error and everything else is being dropped so the retry has
    /// the card. No plan is computed on that path.
    Recovery,
}

impl AdmissionReason {
    /// A stable word for the log, so a run can be filtered by exit rather than
    /// by reading each line. `Budgeted` keeps its own name here even though its
    /// arithmetic is printed separately, because the non-evicting `debug` line
    /// carries the label and nothing else.
    fn label(self) -> &'static str {
        match self {
            Self::Unprofiled => "unprofiled",
            Self::NoTelemetry => "no_telemetry",
            Self::Budgeted(_) => "budgeted",
            Self::Recovery => "recovery",
        }
    }
}

/// The comparison `admission_plan` made, kept whole so a log line can show the
/// reader the same subtraction rather than a verdict.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Budgeted {
    budget_bytes: u64,
    available_bytes: u64,
    safety_reserve_bytes: u64,
    reservation_bytes: u64,
    required_bytes: u64,
    /// Whether the stage's weights were resident when the decision was taken.
    ///
    /// This picks which half of the profile becomes the reservation, and it is
    /// the term that makes an eviction self-sustaining: a resident stage is
    /// charged its incremental `workspace_bytes`, an evicted one its whole
    /// `peak_bytes`. Once a stage has been unloaded, re-admitting it therefore
    /// asks for several times what running it in place asked for.
    loaded: bool,
}

impl Budgeted {
    /// How far short the card fell. Zero on the exit that did not evict.
    fn deficit_bytes(self) -> u64 {
        self.required_bytes.saturating_sub(self.available_bytes)
    }

    /// The larger of the two terms that make up `required_bytes`, named for the
    /// log line.
    ///
    /// Redundant beside the terms themselves, and deliberately so: a chapter run
    /// is triaged by grepping, and `crossed=reservation` answers "was this the
    /// safety margin or the model" without anyone reading a number.
    fn crossed(self) -> &'static str {
        if self.reservation_bytes >= self.safety_reserve_bytes {
            "reservation"
        } else {
            "safety_reserve"
        }
    }

    /// Which half of the profile `reservation_bytes` came from.
    fn reservation_source(self) -> &'static str {
        if self.loaded { "workspace" } else { "peak" }
    }
}

impl Residency {
    pub(crate) fn new(resources: Arc<ResourceMonitor>, release_cached_device_memory: bool) -> Self {
        Self {
            resources,
            lane: Arc::new(tokio::sync::Semaphore::new(1)),
            sequence: AtomicU64::new(1),
            state: Mutex::new(State::default()),
            release_cached_device_memory,
        }
    }

    pub(crate) async fn enter<'a>(&'a self, stage: Stage, stages: &Stages) -> Admission<'a> {
        if self.resources.snapshot().devices.is_empty() {
            return Admission::untracked(stages.loaded(stage));
        }

        let lane = self
            .lane
            .clone()
            .acquire_owned()
            .await
            .expect("accelerator lane is never closed");
        let snapshot = self.resources.snapshot();
        let memory = MemoryBudget::from_snapshot(&snapshot);
        let loaded = stages.loaded(stage);
        let plan = {
            let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            admission_plan(state.profiles.get(&stage).copied(), memory, loaded)
        };
        let clean_profile = plan.profiling && memory.is_some();
        let evicted = if plan.unload_idle {
            self.unload_idle(stage, stages, clean_profile)
        } else {
            Vec::new()
        };
        log_admission(stage, plan.reason, &evicted);
        if !evicted.is_empty() {
            self.settle_resources().await;
        }
        Admission::tracked(self, lane, plan.profiling, stages.loaded(stage))
    }

    pub(crate) async fn recover<'a>(&'a self, stage: Stage, stages: &Stages) -> Admission<'a> {
        if self.resources.snapshot().devices.is_empty() {
            return Admission::untracked(stages.loaded(stage));
        }

        let lane = self
            .lane
            .clone()
            .acquire_owned()
            .await
            .expect("accelerator lane is never closed");
        let evicted = self.unload_idle(stage, stages, false);
        log_admission(stage, AdmissionReason::Recovery, &evicted);
        if !evicted.is_empty() {
            self.settle_resources().await;
        }
        Admission::tracked(self, lane, false, stages.loaded(stage))
    }

    pub(crate) fn observe(
        &self,
        stage: Stage,
        was_loaded: bool,
        measurement: DeviceMemoryMeasurement,
    ) {
        let (Some(budget), Some(before), Some(peak), Some(after)) = (
            measurement.budget_bytes,
            measurement.used_before_bytes,
            measurement.peak_used_bytes,
            measurement.used_after_bytes,
        ) else {
            return;
        };

        let (ratchet, current) = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if was_loaded && !state.profiles.contains_key(&stage) {
                return;
            }

            let profile = state.profiles.entry(stage).or_default();
            let ratchet = profile.observe(was_loaded, budget, before, peak, after);
            (ratchet, *profile)
        };
        // Outside the lock: a `tracing` call runs whatever the process installed
        // as a subscriber, and holding a mutex the whole pipeline contends on
        // across arbitrary formatting work is a bad trade for two fewer lines.
        if let Some(previous) = ratchet {
            log_ratchet(stage, "measurement", previous, current);
        }
    }

    pub(crate) fn penalize(&self, stage: Stage) {
        let Some(memory) = MemoryBudget::from_snapshot(&self.resources.snapshot()) else {
            return;
        };
        let (previous, current) = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            let profile = state.profiles.entry(stage).or_default();
            let previous = *profile;
            let penalty = profile
                .workspace_bytes
                .saturating_div(2)
                .max(memory.budget_bytes.saturating_div(10));
            profile.workspace_bytes = profile.workspace_bytes.saturating_add(penalty);
            profile.peak_bytes = profile
                .peak_bytes
                .saturating_add(penalty)
                .max(
                    profile
                        .resident_bytes
                        .saturating_add(profile.workspace_bytes),
                )
                .min(memory.budget_bytes);
            (previous, *profile)
        };
        // Unconditional, unlike the measurement path: `penalize` is only reached
        // after an out-of-memory failure, so it cannot be noisy, and it moves the
        // same two monotonic numbers by at least a tenth of the card.
        log_ratchet(stage, "oom_penalty", previous, current);
    }

    pub(crate) fn touch(&self, stage: Stage, stages: &Stages) {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        stages.touch(stage, sequence);
    }

    /// Returns the stages whose weights it actually dropped, in the order it
    /// dropped them.
    ///
    /// The victims are the whole story of an eviction and used to be invisible:
    /// each one was announced on its own `debug` line naming only itself, so a
    /// log read after the fact could not tell "the detector released its 153 MiB"
    /// from "the 16 GiB translator was destroyed to make room for it". Handing
    /// the list back lets the caller put all of them on one line, beside the
    /// arithmetic that demanded them.
    fn unload_idle(
        &self,
        requested: Stage,
        stages: &Stages,
        include_requested: bool,
    ) -> Vec<Stage> {
        let mut loaded = Stage::ALL
            .into_iter()
            .filter(|candidate| {
                (include_requested || *candidate != requested) && stages.loaded(*candidate)
            })
            .collect::<Vec<_>>();
        loaded.sort_by_key(|candidate| stages.last_used(*candidate));
        let evicted = loaded
            .into_iter()
            .filter(|candidate| stages.unload(*candidate))
            .collect::<Vec<_>>();
        // Here rather than in the two callers, and before either of them awaits
        // `settle_resources`: dropping a model returns its tensors to torch's
        // caching allocator, not to the driver, so until this runs the sweep has
        // freed nothing that `resources` -- and therefore nothing that
        // `admission_plan` -- can see. Settling first would only re-read the same
        // number and conclude the eviction bought nothing.
        //
        // Safe to call here because `enter` and `recover` both hold the
        // accelerator lane, so no stage is mid-run holding live blocks.
        if !evicted.is_empty() {
            self.empty_cache_if_configured();
        }
        evicted
    }

    /// Hand torch's cached segments back to the driver, if the flag asks for it.
    ///
    /// The ONE gated call both unload paths share. Without it only the eviction
    /// sweep above reached [`empty_torch_cache`], while
    /// `Pipeline::unload_models` -- the `/unload` route and the idle sweep --
    /// dropped the models and left torch's cache holding their segments: a
    /// measured 1.23 GiB residual after an explicit unload. Sharing it is safe:
    /// both explicit-unload callers hold the server's gate, so nothing is
    /// mid-run there either, and `empty_torch_cache` only releases segments
    /// with no live block in any case (never fatal).
    ///
    /// Returns whether the flag asked, so a test can pin the gating without
    /// being able to observe torch.
    pub(crate) fn empty_cache_if_configured(&self) -> bool {
        if self.release_cached_device_memory {
            empty_torch_cache();
        }
        self.release_cached_device_memory
    }

    async fn settle_resources(&self) {
        let mut changed = self.resources.subscribe();
        let _ = tokio::time::timeout(Duration::from_millis(600), changed.changed()).await;
    }
}

impl Admission<'_> {
    fn tracked(
        residency: &Residency,
        lane: tokio::sync::OwnedSemaphorePermit,
        profiling: bool,
        was_loaded: bool,
    ) -> Admission<'_> {
        Admission {
            residency: Some(residency),
            _lane: Some(lane),
            profiling,
            was_loaded,
        }
    }

    fn untracked(was_loaded: bool) -> Self {
        Self {
            residency: None,
            _lane: None,
            profiling: false,
            was_loaded,
        }
    }

    pub(crate) fn tracked_memory(&self) -> bool {
        self.residency.is_some()
    }

    pub(crate) fn profiling(&self) -> bool {
        self.profiling
    }

    pub(crate) fn was_loaded(&self) -> bool {
        self.was_loaded
    }
}

#[derive(Clone, Copy, Debug)]
struct MemoryBudget {
    budget_bytes: u64,
    available_bytes: u64,
}

impl MemoryBudget {
    fn from_snapshot(snapshot: &ResourceSnapshot) -> Option<Self> {
        let device = selected_device(snapshot)?;
        Some(Self {
            budget_bytes: device.memory_budget_bytes?,
            available_bytes: device.memory_available_bytes?,
        })
    }
}

impl ModelProfile {
    /// Folds one measurement in, returning the profile **as it was** if any term
    /// took a new maximum.
    ///
    /// The return value exists only for the log. Every field here is monotonic,
    /// so a ratchet is permanent for the life of the profile, and the caller has
    /// no other way to notice one: `observe` is called after every stage of
    /// every page and is silent on all of them, including the one that raises
    /// the floor for the rest of the run.
    fn observe(
        &mut self,
        loaded: bool,
        budget: u64,
        before: u64,
        peak: u64,
        after: u64,
    ) -> Option<Self> {
        let previous = *self;
        let observed_peak = peak.saturating_sub(before).min(budget);
        if loaded {
            self.workspace_bytes = self.workspace_bytes.max(observed_peak);
        } else {
            let observed_resident = after.saturating_sub(before).min(budget);
            self.resident_bytes = self.resident_bytes.max(observed_resident);
            self.workspace_bytes = self
                .workspace_bytes
                .max(observed_peak.saturating_sub(observed_resident));
        }
        self.peak_bytes = self
            .peak_bytes
            .max(observed_peak)
            .max(self.resident_bytes.saturating_add(self.workspace_bytes))
            .max(reservation_floor(budget));
        (*self != previous).then_some(previous)
    }

    fn reservation(self, loaded: bool, budget: u64) -> u64 {
        let measured = if loaded {
            self.workspace_bytes
        } else {
            self.peak_bytes
        };
        measured.max(reservation_floor(budget)).min(budget)
    }
}

fn admission_plan(
    profile: Option<ModelProfile>,
    memory: Option<MemoryBudget>,
    loaded: bool,
) -> AdmissionPlan {
    let Some(profile) = profile else {
        return AdmissionPlan {
            profiling: true,
            unload_idle: true,
            reason: AdmissionReason::Unprofiled,
        };
    };
    let Some(memory) = memory else {
        return AdmissionPlan {
            profiling: false,
            unload_idle: true,
            reason: AdmissionReason::NoTelemetry,
        };
    };

    let safety_reserve_bytes = safety_reserve(memory.budget_bytes);
    let reservation = profile.reservation(loaded, memory.budget_bytes);
    let required = safety_reserve_bytes.saturating_add(reservation);
    AdmissionPlan {
        profiling: false,
        unload_idle: memory.available_bytes < required,
        reason: AdmissionReason::Budgeted(Budgeted {
            budget_bytes: memory.budget_bytes,
            available_bytes: memory.available_bytes,
            safety_reserve_bytes,
            reservation_bytes: reservation,
            required_bytes: required,
            loaded,
        }),
    }
}

/// Whether an eviction on `device` should end by emptying torch's cache, given
/// what the config asked for.
///
/// Two separate refusals, and they refuse for different reasons. An unset option
/// is upstream's behaviour and has to stay reachable, because how much the call
/// buys is a measurement nobody has taken yet -- `emptyCache` releases a segment
/// only when no live block remains inside it, so a fragmented cache can return a
/// fraction of what it holds while still paying for a device synchronize. A
/// non-CUDA device is refused because there is nothing registered to empty:
/// torch addresses ROCm through the same allocator so it counts, but Metal,
/// Vulkan and CPU do not, and asking would be an FFI call per sweep to be told
/// no.
pub(crate) fn releases_cached_device_memory(
    requested: Option<bool>,
    device: &koharu_ml::Device,
) -> bool {
    requested.unwrap_or(false)
        && matches!(
            device.backend,
            koharu_ml::Backend::Cuda | koharu_ml::Backend::Rocm
        )
}

/// Asks torch to hand its cached-but-unused segments back to the driver.
///
/// Never fatal. The call exists so the *next* admission decision sees the memory
/// this sweep freed; if it fails, the run is exactly as correct as it was before
/// this existed, only slower. Turning that into a panic would kill a page to save
/// a page.
///
/// **All three arms say something, and the middle one is the reason.** The shim
/// reaches the CUDA allocator through `c10::GetAllocator` plus a `dynamic_cast`
/// to `c10::DeviceAllocator`, which is a cross-DLL RTTI match. If that ever stops
/// matching, the shim returns "no allocator" rather than failing, and a silent
/// arm would make an inert fix indistinguishable from a working one in the log of
/// the A/B this flag exists for. A CUDA device that has run a stage always has an
/// allocator, so the warning is unreachable in the healthy case.
fn empty_torch_cache() {
    match koharu_ml::torch::Cuda::empty_cache() {
        // "asked", not "returned": only segments with no live block are released,
        // and nothing here can see how many that was.
        Ok(true) => tracing::debug!("asked torch to return its cached device memory"),
        Ok(false) => tracing::warn!(
            "torch reported no device allocator to empty, so the eviction freed \
             nothing the admission budget can see"
        ),
        Err(error) => {
            tracing::warn!(%error, "could not return cached device memory to the driver");
        }
    }
}

/// Bytes are the currency inside this module, but a log line is read against
/// `nvidia-smi`, which reports MiB. Converting at the edge keeps every number a
/// reader compares in the same unit.
fn mib(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

/// Announces an admission, loudly exactly when it cost something.
///
/// The asymmetry is the whole point. Admission runs four times a page and
/// changes nothing on almost all of them, so the ordinary path stays at `debug`.
/// An eviction is rare and catastrophic -- it is measured at 5,189 ms/page before
/// one and 12,790 ms/page after -- so it goes to `info`, where it lands in an
/// unconfigured run's log and can be found afterwards without having known in
/// advance to turn anything on.
fn log_admission(stage: Stage, reason: AdmissionReason, evicted: &[Stage]) {
    if evicted.is_empty() {
        tracing::debug!(
            stage = %stage,
            reason = reason.label(),
            "admitted stage without evicting"
        );
        return;
    }

    // Joined rather than left as a slice because the fmt subscriber renders a
    // `Debug` slice as `[Translation, Ocr]`, and this field is grepped.
    let victims = evicted
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    // On every eviction line, including the ones carrying no numbers, so that a
    // single `reason=` field partitions a whole chapter. The expected evictions
    // are the first page's `reason=unprofiled` sweeps, one per stage; anything
    // else is a run that has started paying for a cold reload it did not plan.
    let label = reason.label();
    match reason {
        AdmissionReason::Budgeted(budgeted) => tracing::info!(
            stage = %stage,
            evicted = %victims,
            reason = label,
            budget_mib = mib(budgeted.budget_bytes),
            available_mib = mib(budgeted.available_bytes),
            safety_reserve_mib = mib(budgeted.safety_reserve_bytes),
            reservation_mib = mib(budgeted.reservation_bytes),
            reservation_from = budgeted.reservation_source(),
            required_mib = mib(budgeted.required_bytes),
            deficit_mib = mib(budgeted.deficit_bytes()),
            crossed = budgeted.crossed(),
            "evicted resident models to admit a stage"
        ),
        // No arithmetic exists on the other three exits, and printing zeros for
        // the terms would read as "the card was empty" rather than "nobody
        // asked". The label alone says which.
        _ => tracing::info!(
            stage = %stage,
            evicted = %victims,
            reason = label,
            "evicted resident models to admit a stage"
        ),
    }
}

/// Reports a stage's VRAM profile taking a new maximum.
///
/// Every field of `ModelProfile` is monotonic, which makes this the latch: one
/// page carrying an unusually large input raises `peak_bytes` for the life of
/// the process, and `reservation` charges that figure to every later admission
/// of the stage. Nothing else the pipeline emits changes when it happens -- the
/// page that causes it is not slow, only every page after it -- so without this
/// line the mechanism is only visible to an external memory sampler.
fn log_ratchet(stage: Stage, cause: &'static str, previous: ModelProfile, current: ModelProfile) {
    tracing::info!(
        stage = %stage,
        cause,
        resident_mib = mib(current.resident_bytes),
        was_resident_mib = mib(previous.resident_bytes),
        workspace_mib = mib(current.workspace_bytes),
        was_workspace_mib = mib(previous.workspace_bytes),
        peak_mib = mib(current.peak_bytes),
        was_peak_mib = mib(previous.peak_bytes),
        "stage VRAM profile ratcheted"
    );
}

fn reservation_floor(budget: u64) -> u64 {
    budget.saturating_div(100)
}

fn safety_reserve(budget: u64) -> u64 {
    budget
        .saturating_div(10)
        .max(512 * 1024 * 1024)
        .min(budget.saturating_div(3))
}

/// How much VRAM has to be free before `admission_plan` will leave the other
/// stage models resident, given a `reservation` for the stage about to run.
///
/// Public so a caller outside this crate can decide whether a run is worth
/// starting using this crate's own arithmetic, rather than a threshold of its
/// own that would silently drift away from it. Pass `0` for a stage whose
/// weights are already loaded: the clamp lifts it to the floor, which is the
/// least `ModelProfile::reservation` can return for any profile learned here.
#[must_use]
pub fn admission_threshold_bytes(budget: u64, reservation: u64) -> u64 {
    safety_reserve(budget).saturating_add(reservation.max(reservation_floor(budget)).min(budget))
}

pub(crate) fn is_out_of_memory(error: &anyhow::Error) -> bool {
    error.chain().any(|source| {
        let message = source.to_string().to_ascii_lowercase();
        message.contains("out of memory")
            || message.contains("cuda_error_out_of_memory")
            || message.contains("not enough memory")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    fn profile(resident: u64, workspace: u64) -> ModelProfile {
        ModelProfile {
            resident_bytes: resident,
            workspace_bytes: workspace,
            peak_bytes: resident + workspace,
        }
    }

    fn memory(available: u64) -> Option<MemoryBudget> {
        budget_of(8 * GIB, available)
    }

    fn budget_of(budget: u64, available: u64) -> Option<MemoryBudget> {
        Some(MemoryBudget {
            budget_bytes: budget,
            available_bytes: available,
        })
    }

    /// The arithmetic an eviction is expected to have logged. Panics rather than
    /// returning an `Option`, because every caller here has already asserted
    /// which exit was taken.
    fn budgeted(plan: AdmissionPlan) -> Budgeted {
        match plan.reason {
            AdmissionReason::Budgeted(budgeted) => budgeted,
            other => panic!("expected a budgeted admission, got {other:?}"),
        }
    }

    #[test]
    fn unknown_models_are_profiled_exclusively() {
        assert_eq!(
            admission_plan(None, memory(6 * GIB), false),
            AdmissionPlan {
                profiling: true,
                unload_idle: true,
                reason: AdmissionReason::Unprofiled,
            }
        );
    }

    #[test]
    fn loaded_models_reserve_only_incremental_workspace() {
        let plan = admission_plan(Some(profile(3 * GIB, GIB)), memory(3 * GIB), true);
        assert!(!plan.profiling);
        assert!(!plan.unload_idle);
        assert_eq!(budgeted(plan).reservation_bytes, GIB);
    }

    #[test]
    fn insufficient_capacity_evicts_idle_models_before_running() {
        let plan = admission_plan(Some(profile(2 * GIB, GIB)), memory(3 * GIB), false);
        assert!(!plan.profiling);
        assert!(plan.unload_idle);
    }

    #[test]
    fn missing_telemetry_evicts_idle_models() {
        assert_eq!(
            admission_plan(Some(profile(GIB, GIB)), None, false),
            AdmissionPlan {
                profiling: false,
                unload_idle: true,
                reason: AdmissionReason::NoTelemetry,
            }
        );
    }

    #[test]
    fn profile_separates_residency_from_incremental_workspace() {
        let mut measured = ModelProfile::default();
        assert!(
            measured
                .observe(false, 8 * GIB, GIB, 5 * GIB, 4 * GIB)
                .is_some()
        );
        assert_eq!(measured.resident_bytes, 3 * GIB);
        assert_eq!(measured.workspace_bytes, GIB);
        assert_eq!(measured.peak_bytes, 4 * GIB);

        assert!(
            measured
                .observe(true, 8 * GIB, 4 * GIB, 6 * GIB, 4 * GIB)
                .is_some()
        );
        assert_eq!(measured.workspace_bytes, 2 * GIB);
        assert_eq!(measured.peak_bytes, 5 * GIB);
    }

    #[test]
    fn an_unset_release_option_leaves_the_torch_cache_exactly_as_upstream_left_it() {
        assert!(!releases_cached_device_memory(
            None,
            &koharu_ml::Device::cuda(0)
        ));
    }

    #[test]
    fn a_cuda_device_asked_to_release_its_cache_does() {
        assert!(releases_cached_device_memory(
            Some(true),
            &koharu_ml::Device::cuda(0)
        ));
    }

    /// The one gated call both unload paths share, driven through
    /// the method the two call sites actually call rather than the flag
    /// resolution beside it. Only the OFF arm can run here: the ON arm calls
    /// into the torch shim, whose lazy DLL load panics in this crate's
    /// un-wrapped test gate -- which makes this single assertion a tripwire in
    /// BOTH directions. A method hardwired to `true` returns the wrong value
    /// AND panics on that very load, right here. The ON arm's evidence is a
    /// measurement: the post-`/unload` VRAM residual, which no unit test can
    /// observe.
    #[test]
    fn the_shared_cache_empty_never_asks_when_the_flag_says_no() {
        let resources = crate::resources::ResourceMonitor::new(&koharu_ml::Device::cpu());
        assert!(!Residency::new(resources, false).empty_cache_if_configured());
    }

    #[test]
    fn an_explicit_false_is_the_off_arm_of_the_measurement_and_not_the_default() {
        // Both arms have to be reachable from the config alone, or the two runs
        // being compared differ by a rebuild rather than by a flag.
        assert!(!releases_cached_device_memory(
            Some(false),
            &koharu_ml::Device::cuda(0)
        ));
    }

    #[test]
    fn rocm_releases_because_torch_addresses_it_through_the_same_allocator() {
        // `koharu_torch::Device::try_from` maps Rocm onto Cuda, which is the
        // assumption every other CUDA entry point in the shim already makes.
        assert!(releases_cached_device_memory(
            Some(true),
            &koharu_ml::Device::rocm(0)
        ));
    }

    #[test]
    fn a_cpu_run_never_asks_because_there_is_no_device_cache_to_empty() {
        assert!(!releases_cached_device_memory(
            Some(true),
            &koharu_ml::Device::cpu()
        ));
    }

    #[test]
    fn vulkan_and_metal_are_refused_rather_than_asked_and_told_no() {
        // Neither registers a caching allocator under `DeviceType::CUDA`, so the
        // shim would return "no allocator" on every sweep. Cheaper to not ask.
        assert!(!releases_cached_device_memory(
            Some(true),
            &koharu_ml::Device::vulkan(0)
        ));
        assert!(!releases_cached_device_memory(
            Some(true),
            &koharu_ml::Device::metal(0)
        ));
    }

    #[test]
    fn an_eviction_carries_the_subtraction_that_caused_it() {
        let plan = admission_plan(Some(profile(2 * GIB, GIB)), memory(3 * GIB), false);
        assert!(plan.unload_idle);

        let budgeted = budgeted(plan);
        assert_eq!(budgeted.budget_bytes, 8 * GIB);
        assert_eq!(budgeted.available_bytes, 3 * GIB);
        assert_eq!(
            budgeted.required_bytes,
            budgeted.safety_reserve_bytes + budgeted.reservation_bytes
        );
        assert_eq!(
            budgeted.deficit_bytes(),
            budgeted.required_bytes - budgeted.available_bytes
        );
        assert!(budgeted.deficit_bytes() > 0);
    }

    #[test]
    fn an_admission_that_evicts_nothing_still_carries_its_arithmetic() {
        // The non-evicting exit is logged at debug, but it is the same exit and
        // has to answer "how close was it" when the next page does evict.
        let plan = admission_plan(Some(profile(3 * GIB, GIB)), memory(3 * GIB), true);
        assert!(!plan.unload_idle);
        assert_eq!(budgeted(plan).deficit_bytes(), 0);
    }

    #[test]
    fn an_unloaded_stage_is_charged_its_whole_peak_rather_than_its_workspace() {
        // The measured shape of the latch on a 32 GiB card: a translator whose
        // incremental workspace is ~2.3 GiB but whose whole-model peak is 19 GiB.
        // Nothing about the card changes between these two calls -- only whether
        // the weights are still resident -- and that alone moves the requirement
        // from "fits in a warm 6 GiB" to "needs more than two thirds of the card".
        let learned = profile(17_101 * MIB, 2_355 * MIB);
        let warm = admission_plan(Some(learned), budget_of(32 * GIB, 6 * GIB), true);
        let cold = admission_plan(Some(learned), budget_of(32 * GIB, 6 * GIB), false);

        assert!(!warm.unload_idle);
        assert!(cold.unload_idle);
        assert_eq!(budgeted(warm).reservation_source(), "workspace");
        assert_eq!(budgeted(cold).reservation_source(), "peak");
        assert!(
            budgeted(cold).reservation_bytes >= 8 * budgeted(warm).reservation_bytes,
            "the peak-sourced reservation should be the ~8x jump the log has to make visible"
        );
    }

    #[test]
    fn the_crossed_term_names_whichever_half_of_the_requirement_is_larger() {
        let dominated_by_the_model = Budgeted {
            budget_bytes: 32 * GIB,
            available_bytes: GIB,
            safety_reserve_bytes: 3 * GIB,
            reservation_bytes: 19 * GIB,
            required_bytes: 22 * GIB,
            loaded: false,
        };
        assert_eq!(dominated_by_the_model.crossed(), "reservation");

        let dominated_by_the_margin = Budgeted {
            reservation_bytes: 100 * MIB,
            required_bytes: 3 * GIB + 100 * MIB,
            loaded: true,
            ..dominated_by_the_model
        };
        assert_eq!(dominated_by_the_margin.crossed(), "safety_reserve");
    }

    #[test]
    fn every_admission_exit_has_its_own_label() {
        // The labels are the grep contract for a chapter run, so a rename is a
        // breaking change to how these logs are read, not a cosmetic edit.
        let labels = [
            AdmissionReason::Unprofiled.label(),
            AdmissionReason::NoTelemetry.label(),
            AdmissionReason::Recovery.label(),
            AdmissionReason::Budgeted(Budgeted {
                budget_bytes: GIB,
                available_bytes: GIB,
                safety_reserve_bytes: 0,
                reservation_bytes: 0,
                required_bytes: 0,
                loaded: true,
            })
            .label(),
        ];
        // `to_vec` because `dedup` is a Vec method, not a slice one. Without it
        // this neither compiles nor asserts anything: an array cannot shrink, so
        // the comparison below would be 4 == 4 whatever the labels turned out to be.
        let mut sorted = labels.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len());
    }

    #[test]
    fn the_first_measurement_of_a_stage_reports_a_ratchet_from_nothing() {
        let mut measured = ModelProfile::default();
        let previous = measured
            .observe(false, 8 * GIB, GIB, 5 * GIB, 4 * GIB)
            .expect("the first measurement always raises every term from zero");
        assert_eq!(previous, ModelProfile::default());
    }

    #[test]
    fn a_second_page_of_the_same_size_reports_no_ratchet() {
        // The quiet case, and the reason this can be logged at info at all: an
        // ordinary chapter measures the same shape over and over and says
        // nothing, so a line in the log means the floor really did move.
        let mut measured = ModelProfile::default();
        let _ = measured.observe(false, 8 * GIB, GIB, 5 * GIB, 4 * GIB);
        assert_eq!(
            measured.observe(false, 8 * GIB, GIB, 5 * GIB, 4 * GIB),
            None
        );
    }

    #[test]
    fn an_oversized_page_reports_the_workspace_it_replaced() {
        let mut measured = ModelProfile::default();
        let _ = measured.observe(false, 8 * GIB, GIB, 5 * GIB, 4 * GIB);
        assert_eq!(measured.workspace_bytes, GIB);

        // One page whose largest crop is ~60x a text region: the workspace
        // trebles and never comes back down.
        let previous = measured
            .observe(true, 8 * GIB, 4 * GIB, 7 * GIB, 4 * GIB)
            .expect("a larger transient allocation raises the workspace");
        assert_eq!(previous.workspace_bytes, GIB);
        assert_eq!(measured.workspace_bytes, 3 * GIB);
        assert_eq!(previous.peak_bytes, 4 * GIB);
        assert_eq!(measured.peak_bytes, 6 * GIB);
    }
}
