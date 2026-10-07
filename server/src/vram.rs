//! What the server can honestly say about VRAM, and the threshold it refuses on.
//!
//! The arithmetic is koharu's own: `admission_threshold_bytes` is the
//! `safety_reserve + reservation` comparison `admission_plan` makes, exported from
//! koharu-pipeline rather than reimplemented here, so this cannot drift away from the
//! decision the pipeline itself will take.
//!
//! Two things about that comparison have to be said out loud, because the name
//! `sufficient` overpromises otherwise.
//!
//! * Koharu has no refusal path at all. `admission_plan` only chooses between
//!   evicting the other stage models and leaving them resident; on the cold path,
//!   where no stage has been profiled yet, it does not consult memory. So this
//!   answers "would koharu keep the other models resident", which is a good proxy
//!   for "this will run without thrashing" -- but the refusal built on top of it
//!   is BireLate's, not koharu's.
//! * The cold half of the estimate is a guess. Koharu learns the real figure as
//!   a stage's measured peak during a run, and that number is private and only
//!   exists after the run it would have protected. `--cold-reserve-bytes` is the
//!   escape hatch.
//!
//! On Windows the numbers are DXGI's, which are *per process*: `budget_bytes` is
//! the budget the OS assigned this process, and the used figure behind
//! `available_bytes` counts only this process's allocations. Memory another
//! process holds -- an Ollama model, say -- never appears as "used"; it appears
//! as a smaller budget. That is the right input for
//! an admission decision and the wrong thing to label "VRAM used / total card".

use koharu_pipeline::{ResourceSnapshot, admission_threshold_bytes, selected_device};
use koharu_translator::Provider;
use tokio::sync::watch;

pub const GIB: u64 = 1024 * 1024 * 1024;

/// Detection, OCR and inpainting together, cold. A guess: see the module
/// comment. Deliberately generous, because being wrong in the other direction
/// costs a native `abort()` with no Rust panic and no message.
pub const COLD_RESERVE_STAGES: u64 = 4 * GIB;

/// What koharu's bundled llama.cpp engine wants for the default
/// `gemma4-26b-a4b-it`, which is the model `--provider local` pins unless told
/// otherwise. Not charged under `ollama`: those weights live in Ollama's address
/// space, which is the whole reason that provider is the recommended one.
///
/// **17 GiB.** The model's weights are **13.26 GB**, measured. On top of the
/// weights this covers ~1.5 GiB of KV at the peak `n_ctx` the shipping 96-pair
/// story window reaches (3,840 tokens measured, at 220 KiB of KV per token)
/// plus the ~537 MiB compute plateau, then rounds up. That leaves ~3.7 of
/// slack over the weights -- deliberate generosity, for the reason in the
/// module comment: being wrong in the other direction costs a native `abort()`.
///
/// **It is one number, not a per-model table.** Pinning a model larger than the
/// default under-reserves: `--llm gemma4-31b-it` wants roughly 3 GB more than
/// this, so pass `--cold-reserve-bytes` with it.
pub const COLD_RESERVE_LOCAL_LLM: u64 = 17 * GIB;

/// What one page needs before anything has been loaded.
#[must_use]
pub fn default_cold_reserve(provider: Provider) -> u64 {
    match provider {
        Provider::Local => COLD_RESERVE_STAGES + COLD_RESERVE_LOCAL_LLM,
        _ => COLD_RESERVE_STAGES,
    }
}

/// Three states collapsed into two, because only one distinction changes an
/// answer: either there are numbers to compare, or there are not.
///
/// `Unknown` covers a monitor that has not published yet (the watch channel
/// starts at `ResourceSnapshot::default()`, whose device list is *empty* rather
/// than a device reporting zeros), a `--cpu` server (which reports no devices at
/// all), and a device whose DXGI/NVML query failed (which reports `None` for
/// every metric, never `0`). None of those mean "no memory free", and koharu
/// treats the same absence as "run untracked" rather than as a reason to refuse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Vram {
    Unknown,
    Known {
        device: String,
        budget_bytes: u64,
        available_bytes: u64,
    },
}

/// One reading and the verdict drawn from it, so a handler cannot report a
/// `sufficient` that was computed against a different sample than its `vram`.
#[derive(Clone, Debug)]
pub struct Headroom {
    pub vram: Vram,
    /// What `sufficient` compared against. Zero when there is nothing to compare.
    pub needed_bytes: u64,
    pub sufficient: bool,
}

/// Reads the monitor's latest publication.
///
/// Cloned rather than borrowed: `borrow` returns a guard holding the watch
/// channel's read lock, and that must not survive into an `async fn`'s future.
/// The clone is a handful of short strings.
#[must_use]
pub fn headroom(
    receiver: &watch::Receiver<ResourceSnapshot>,
    warm: bool,
    cold_reserve_bytes: u64,
) -> Headroom {
    let snapshot = receiver.borrow().clone();
    assess(&snapshot, warm, cold_reserve_bytes)
}

#[must_use]
pub fn assess(snapshot: &ResourceSnapshot, warm: bool, cold_reserve_bytes: u64) -> Headroom {
    let vram = read(snapshot);
    match &vram {
        // Permissive on purpose: refusing here would 507 every request in the
        // milliseconds before the first sample lands, and would brick a --cpu
        // server outright.
        Vram::Unknown => Headroom {
            vram,
            needed_bytes: 0,
            sufficient: true,
        },
        Vram::Known {
            budget_bytes,
            available_bytes,
            ..
        } => {
            let needed_bytes = needed(*budget_bytes, warm, cold_reserve_bytes);
            let sufficient = *available_bytes >= needed_bytes;
            Headroom {
                needed_bytes,
                sufficient,
                vram,
            }
        }
    }
}

/// The reading a page that *changes* the models has to satisfy.
///
/// `assess`'s warm arm is unsound for such a page, and the way it fails is
/// silent. `Pipeline::reload` builds a fresh `StageRunner`, and with it a fresh
/// `Residency` whose profile map is empty. `admission_plan` answers
/// `{ profiling: true, unload_idle: true }` for a stage it has no profile for,
/// and `unload_idle` is then called with `include_requested`, so it evicts every
/// loaded model -- the one being admitted included, so it can be measured alone.
/// `Stage::Translation`'s unload is the local LLM's. So the first stage after a
/// reload destroys every warm weight on the card, the ~13 GB model included, and
/// the rest of that page is a cold start. A request asking for `warm` there is
/// exempted from the cold check by weights it is about to throw away.
///
/// The occupancy is the wrong denominator too, and in the other direction: those
/// bytes come back. On Windows the used figure counts only this process's own
/// allocations (see the module comment), so once the eviction has run,
/// availability *is* the budget. What has to hold is therefore that the budget
/// could host a cold start at all -- which is the question of what the other
/// processes on the card have left us, and is exactly the shape this refusal
/// exists for: Ollama or another program taking the card while our weights sat
/// resident, shrinking the budget under them.
///
/// Since `available <= budget`, this is strictly weaker than the cold-start test
/// a stone-cold server takes, so it can never refuse a page a cold server would
/// have accepted. Charging the cold reserve against the *current* availability
/// instead would 507 every model change on a healthy warm card, whose whole
/// budget is by then correctly spoken for.
#[must_use]
pub fn after_reload(
    receiver: &watch::Receiver<ResourceSnapshot>,
    cold_reserve_bytes: u64,
) -> Headroom {
    let snapshot = receiver.borrow().clone();
    assess_after_reload(&snapshot, cold_reserve_bytes)
}

#[must_use]
pub fn assess_after_reload(snapshot: &ResourceSnapshot, cold_reserve_bytes: u64) -> Headroom {
    match read(snapshot) {
        // Permissive for the same reasons `assess` is: no sample yet, --cpu, or
        // a failed query. None of them mean "no memory".
        vram @ Vram::Unknown => Headroom {
            vram,
            needed_bytes: 0,
            sufficient: true,
        },
        Vram::Known {
            device,
            budget_bytes,
            ..
        } => {
            let needed_bytes = needed(budget_bytes, false, cold_reserve_bytes);
            Headroom {
                sufficient: budget_bytes >= needed_bytes,
                needed_bytes,
                vram: Vram::Known {
                    device,
                    budget_bytes,
                    // The post-eviction figure, written here rather than carried
                    // alongside so `shortfall` measures the gap that will
                    // actually exist and the 507 names that number. Deliberately
                    // not a reading of the card as it stands: this Headroom
                    // describes the card the page will run on, and it must not
                    // be reported as telemetry.
                    available_bytes: budget_bytes,
                },
            }
        }
    }
}

/// How much has to be free for the run to be worth starting.
///
/// A warm stage reserves nothing extra, so `0` is passed through and koharu's
/// own clamp lifts it to the floor no profile ever goes below. A cold one has no
/// measured figure to pass, hence the estimate.
#[must_use]
pub fn needed(budget_bytes: u64, warm: bool, cold_reserve_bytes: u64) -> u64 {
    let reservation = if warm { 0 } else { cold_reserve_bytes };
    admission_threshold_bytes(budget_bytes, reservation)
}

/// How far short the reading is, in bytes. Zero when there is no shortfall or
/// nothing to measure.
#[must_use]
pub fn shortfall(headroom: &Headroom) -> u64 {
    match &headroom.vram {
        Vram::Unknown => 0,
        Vram::Known {
            available_bytes, ..
        } => headroom.needed_bytes.saturating_sub(*available_bytes),
    }
}

/// Powers of 1024, spelled "GB", because that is how the other memory readouts
/// this is read beside label it too.
#[must_use]
pub fn gb(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

fn read(snapshot: &ResourceSnapshot) -> Vram {
    let Some(device) = selected_device(snapshot) else {
        return Vram::Unknown;
    };
    match (device.memory_budget_bytes, device.memory_available_bytes) {
        // A zero budget is koharu's own "the query returned nothing useful"
        // marker, and dividing a threshold out of it would make every request
        // look fine.
        (Some(budget_bytes), Some(available_bytes)) if budget_bytes > 0 => Vram::Known {
            device: device.name.clone(),
            budget_bytes,
            available_bytes,
        },
        _ => Vram::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use koharu_pipeline::DeviceResources;

    use super::*;

    fn device(name: &str, budget: Option<u64>, available: Option<u64>) -> DeviceResources {
        DeviceResources {
            name: name.to_owned(),
            selected: true,
            memory_budget_bytes: budget,
            memory_used_bytes: budget
                .zip(available)
                .map(|(budget, available)| budget.saturating_sub(available)),
            memory_available_bytes: available,
            utilization_percent: None,
        }
    }

    fn snapshot(devices: Vec<DeviceResources>) -> ResourceSnapshot {
        ResourceSnapshot {
            devices,
            ..ResourceSnapshot::default()
        }
    }

    #[test]
    fn no_sample_yet_is_not_an_empty_card() {
        // The watch channel starts at the default, whose device list is empty.
        // Reading that as "0 bytes free" would 507 every request for the first
        // few milliseconds of the process.
        let headroom = assess(&ResourceSnapshot::default(), false, 24 * GIB);
        assert_eq!(headroom.vram, Vram::Unknown);
        assert!(headroom.sufficient);
        assert_eq!(headroom.needed_bytes, 0);
        assert_eq!(shortfall(&headroom), 0);
    }

    #[test]
    fn a_device_without_telemetry_is_also_unknown() {
        // Both shapes koharu produces when the query fails: every metric None,
        // and a budget that read back as zero.
        for absent in [
            device("Cuda0", None, None),
            device("NVIDIA GeForce RTX 5090", Some(0), Some(0)),
        ] {
            let headroom = assess(&snapshot(vec![absent]), false, 24 * GIB);
            assert_eq!(headroom.vram, Vram::Unknown);
            assert!(headroom.sufficient);
        }
    }

    #[test]
    fn the_selected_device_is_the_one_reported() {
        let mut first = device("Intel UHD", Some(2 * GIB), Some(GIB));
        first.selected = false;
        let headroom = assess(
            &snapshot(vec![
                first,
                device("NVIDIA GeForce RTX 5090", Some(32 * GIB), Some(30 * GIB)),
            ]),
            true,
            0,
        );
        assert_eq!(
            headroom.vram,
            Vram::Known {
                device: "NVIDIA GeForce RTX 5090".to_owned(),
                budget_bytes: 32 * GIB,
                available_bytes: 30 * GIB,
            }
        );
    }

    #[test]
    fn the_threshold_is_koharus_own_arithmetic() {
        // safety_reserve(32 GiB) = max(3.2 GiB, 512 MiB) capped at 10.6 GiB, and
        // a warm stage reserves only the floor, 32 GiB / 100.
        let budget = 32 * GIB;
        let warm = needed(budget, true, 24 * GIB);
        assert_eq!(warm, budget / 10 + budget / 100);
        // Cold adds the estimate instead of the floor.
        assert_eq!(needed(budget, false, 24 * GIB), budget / 10 + 24 * GIB);
        // The estimate is clamped to the budget, exactly as a profile would be.
        assert_eq!(needed(budget, false, 999 * GIB), budget / 10 + budget);
        // An estimate below the floor is lifted to it.
        assert_eq!(needed(budget, false, 1), warm);
    }

    #[test]
    fn a_warm_server_with_room_says_yes_and_a_cold_one_does_not() {
        // Both readings are the same card with 6 GB free: enough to keep the
        // resident models where they are, nowhere near enough to load a 31B LLM.
        let reading = snapshot(vec![device(
            "NVIDIA GeForce RTX 5090",
            Some(32 * GIB),
            Some(6 * GIB),
        )]);
        assert!(assess(&reading, true, 24 * GIB).sufficient);

        let cold = assess(&reading, false, 24 * GIB);
        assert!(!cold.sufficient);
        assert_eq!(cold.needed_bytes, 32 * GIB / 10 + 24 * GIB);
        assert_eq!(shortfall(&cold), cold.needed_bytes - 6 * GIB);
    }

    #[test]
    fn a_card_the_process_has_to_itself_admits_a_cold_run() {
        let reading = snapshot(vec![device(
            "NVIDIA GeForce RTX 5090",
            Some(32 * GIB),
            Some(31 * GIB),
        )]);
        assert!(assess(&reading, false, 24 * GIB).sufficient);
    }

    #[test]
    fn the_local_engine_is_charged_for_its_own_weights_and_ollama_is_not() {
        // The whole point of --provider ollama: those weights are already in
        // another process, so they must not be budgeted for twice.
        assert_eq!(
            default_cold_reserve(Provider::Local),
            COLD_RESERVE_STAGES + COLD_RESERVE_LOCAL_LLM
        );
        assert_eq!(
            default_cold_reserve(Provider::OpenAiCompatible),
            COLD_RESERVE_STAGES
        );
    }

    #[test]
    fn a_reload_is_judged_on_the_budget_not_on_what_we_are_holding() {
        // The ordinary warm server: 26 GB of its own weights resident, so only
        // 6 GB reads as free. Those bytes come back when the reload's first
        // stage evicts everything, so this must not be refused -- charging the
        // cold reserve against the 6 GB would 507 every model change.
        let healthy = snapshot(vec![device(
            "NVIDIA GeForce RTX 5090",
            Some(32 * GIB),
            Some(6 * GIB),
        )]);
        assert!(assess_after_reload(&healthy, 24 * GIB).sufficient);

        // The same 6 GB free, but the budget itself has shrunk -- another
        // process took the card while our weights sat resident. The eviction
        // hands back everything we hold and it still is not enough, so the
        // reload would abort with no Rust error at all.
        let squeezed = snapshot(vec![device(
            "NVIDIA GeForce RTX 5090",
            Some(12 * GIB),
            Some(6 * GIB),
        )]);
        let refused = assess_after_reload(&squeezed, 24 * GIB);
        assert!(!refused.sufficient);
        // The cold reserve is clamped to the budget, exactly as `needed` does.
        assert_eq!(refused.needed_bytes, 12 * GIB / 10 + 12 * GIB);
        assert_eq!(shortfall(&refused), 12 * GIB / 10);
    }

    #[test]
    fn a_reload_can_never_be_refused_where_a_cold_start_would_be_admitted() {
        // available <= budget, so this test is weaker than the cold one by
        // construction. Asserted rather than argued, because the day it stops
        // holding is the day a model change starts 507-ing a card with room.
        for available in [0, GIB, 6 * GIB, 27 * GIB, 31 * GIB, 32 * GIB] {
            let reading = snapshot(vec![device(
                "NVIDIA GeForce RTX 5090",
                Some(32 * GIB),
                Some(available),
            )]);
            for reserve in [4 * GIB, 24 * GIB, 999 * GIB] {
                if assess(&reading, false, reserve).sufficient {
                    assert!(
                        assess_after_reload(&reading, reserve).sufficient,
                        "{available} free, {reserve} reserved"
                    );
                }
            }
        }
    }

    #[test]
    fn a_reload_with_no_telemetry_is_not_refused() {
        // Same rule as `assess`: a --cpu server and the first milliseconds of
        // the process must not 507.
        for absent in [
            ResourceSnapshot::default(),
            snapshot(vec![device("Cuda0", None, None)]),
        ] {
            let headroom = assess_after_reload(&absent, 24 * GIB);
            assert_eq!(headroom.vram, Vram::Unknown);
            assert!(headroom.sufficient);
            assert_eq!(headroom.needed_bytes, 0);
        }
    }

    #[test]
    fn an_absurd_reading_neither_panics_nor_wraps() {
        let reading = snapshot(vec![device(
            "NVIDIA GeForce RTX 5090",
            Some(u64::MAX),
            Some(0),
        )]);
        let headroom = assess(&reading, false, u64::MAX);
        assert!(!headroom.sufficient);
        assert_eq!(shortfall(&headroom), headroom.needed_bytes);
    }
}
