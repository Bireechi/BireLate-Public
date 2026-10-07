//! Koharu's downloads, on the console. The first start fetches gigabytes of
//! runtimes before the listener binds, and the first page gigabytes of models;
//! without this both look like a hang.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use koharu_runtime::downloads::{Event, subscribe};
use tokio::sync::broadcast::error::RecvError;

/// A progress line needs BOTH, so neither a fast nor a slow file floods the
/// log: five seconds since the file's last line, and ten more percent...
const INTERVAL: Duration = Duration::from_secs(5);
const STEP_PERCENT: u64 = 10;
/// ...unless the file has been quiet this long: ten percent of a 13 GB model
/// over a slow link is minutes.
const MAX_QUIET: Duration = Duration::from_secs(30);

/// Spawns the logger. Call it before anything downloads: a receiver only sees
/// events sent after it subscribed.
pub fn log_progress() {
    let mut events = subscribe();
    tokio::spawn(async move {
        let mut files = Files::default();
        loop {
            match events.recv().await {
                Ok(event) => match files.line(&event, Instant::now()) {
                    Some(line) if matches!(event, Event::Failed { .. }) => tracing::warn!("{line}"),
                    Some(line) => tracing::info!("{line}"),
                    None => {}
                },
                // Koharu publishes a `Progress` per chunk, so falling behind is
                // routine; the next one carries the running total anyway.
                Err(RecvError::Lagged(_)) => {}
                Err(RecvError::Closed) => break,
            }
        }
    });
}

struct File {
    name: String,
    /// When this file's last progress line was logged, and at what percent;
    /// `None` until its first `Progress`, which is always logged.
    logged: Option<(Instant, u64)>,
}

/// In-flight files by download id. `Finished` carries only the id, so the
/// name has to be remembered from the events before it.
#[derive(Default)]
struct Files(HashMap<u64, File>);

impl Files {
    fn line(&mut self, event: &Event, now: Instant) -> Option<String> {
        match event {
            Event::Started { id, name } => {
                // At once: the first `Progress` (and the size it carries) can
                // be a whole part away, which is minutes on a big file.
                self.0.insert(*id, File { name: name.clone(), logged: None });
                Some(format!("downloading {name}"))
            }
            Event::Progress { id, name, completed, total } => {
                // `or_insert` as well as `Started`, in case that was lost to lag.
                let file = self.0.entry(*id).or_insert_with(|| File {
                    name: name.clone(),
                    logged: None,
                });
                // `total` is 0 when the server sent no length: time alone gates.
                let percent = (*total > 0).then(|| completed.saturating_mul(100) / total);
                if let Some((at, last)) = file.logged {
                    let quiet = now.duration_since(at);
                    let due = quiet >= MAX_QUIET
                        || (quiet >= INTERVAL && percent.is_none_or(|p| p >= last + STEP_PERCENT));
                    if !due {
                        return None;
                    }
                }
                file.logged = Some((now, percent.unwrap_or(0)));
                Some(match percent {
                    Some(p) => format!("{name}: {} of {} ({p}%)", mb(*completed), mb(*total)),
                    None => format!("{name}: {}", mb(*completed)),
                })
            }
            Event::Finished { id } => Some(format!("downloaded {}", self.0.remove(id)?.name)),
            Event::Failed { id, name, error } => {
                self.0.remove(id);
                Some(format!("download of {name} failed: {error}"))
            }
        }
    }
}

fn mb(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(completed: u64, total: u64) -> Event {
        Event::Progress { id: 1, name: "llama.zip".to_owned(), completed, total }
    }

    #[test]
    fn a_file_logs_its_start_throttled_progress_and_its_end() {
        let t0 = Instant::now();
        let at = |secs| t0 + Duration::from_secs(secs);
        let mut files = Files::default();
        let total = 200_000_000;

        // The start line goes out at once, before any size is known.
        assert_eq!(
            files.line(&Event::Started { id: 1, name: "llama.zip".to_owned() }, t0).as_deref(),
            Some("downloading llama.zip")
        );
        // The first `Progress` is never throttled: it carries the size.
        assert_eq!(
            files.line(&progress(10_000_000, total), at(0)).as_deref(),
            Some("llama.zip: 10.0 MB of 200.0 MB (5%)")
        );
        // Six seconds but only seven points (5% -> 12%): too little.
        assert_eq!(files.line(&progress(24_000_000, total), at(6)), None);
        assert_eq!(
            files.line(&progress(40_000_000, total), at(7)).as_deref(),
            Some("llama.zip: 40.0 MB of 200.0 MB (20%)")
        );
        // Two points in 29 s is still too little; three in 30 s is due anyway.
        assert_eq!(files.line(&progress(44_000_000, total), at(36)), None);
        assert_eq!(
            files.line(&progress(46_000_000, total), at(37)).as_deref(),
            Some("llama.zip: 46.0 MB of 200.0 MB (23%)")
        );
        // Sixty-seven points but one second: too soon.
        assert_eq!(files.line(&progress(180_000_000, total), at(38)), None);
        assert_eq!(
            files.line(&Event::Finished { id: 1 }, at(39)).as_deref(),
            Some("downloaded llama.zip")
        );
        // Forgotten once finished: a stray repeat says nothing.
        assert_eq!(files.line(&Event::Finished { id: 1 }, at(39)), None);
    }

    #[test]
    fn an_unknown_size_lost_start_and_failure_still_log() {
        let t0 = Instant::now();
        let mut files = Files::default();
        // No `Started` (lost to lag) and no length from the server.
        // The first `Progress` still logs, size or no size.
        assert_eq!(files.line(&progress(1_000_000, 0), t0).as_deref(), Some("llama.zip: 1.0 MB"));
        assert_eq!(files.line(&progress(2_000_000, 0), t0 + Duration::from_secs(4)), None);
        assert_eq!(
            files.line(&progress(3_000_000, 0), t0 + Duration::from_secs(5)).as_deref(),
            Some("llama.zip: 3.0 MB")
        );
        let failed = Event::Failed { id: 1, name: "llama.zip".to_owned(), error: "reset".to_owned() };
        assert_eq!(files.line(&failed, t0).as_deref(), Some("download of llama.zip failed: reset"));
        assert_eq!(files.line(&Event::Finished { id: 1 }, t0), None);
    }
}
