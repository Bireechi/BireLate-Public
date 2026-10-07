use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use koharu_scene::{EntityId, Patch};

use crate::{
    ErrorKind, PipelineConfig, PipelineError, Progress, ProgressSink, Stage, StopToken, progress,
    residency::{Admission, Residency, is_out_of_memory, releases_cached_device_memory},
    resources::ResourceMonitor,
    stages::{StageInput, Stages},
};

pub(crate) struct StageRunner {
    stages: Stages,
    residency: Residency,
    resources: Arc<ResourceMonitor>,
}

impl StageRunner {
    pub(crate) fn new(
        config: &PipelineConfig,
        translator: koharu_translator::Translator,
        device: &koharu_ml::Device,
        resources: Arc<ResourceMonitor>,
    ) -> Result<Self> {
        // Resolved here rather than inside `Residency` because this is the only
        // place holding both halves of the answer: the config that asked for it,
        // and the device the stages were actually built on.
        let release_cached_device_memory =
            releases_cached_device_memory(config.processor.release_cached_vram, device);
        Ok(Self {
            stages: Stages::new(config, translator, device)?,
            residency: Residency::new(resources.clone(), release_cached_device_memory),
            resources,
        })
    }

    /// Loads `stage`'s weights without running anything.
    pub(crate) async fn load(&self, stage: Stage) -> Result<()> {
        self.stages.load(stage).await
    }

    /// Drops `stage`'s weights while keeping this runner alive.
    ///
    /// The models hang off `stages` and the learned VRAM profiles off
    /// `residency`, which are sibling fields, so dropping a model cannot touch a
    /// profile. Rebuilding the runner to free memory throws away both.
    pub(crate) fn unload(&self, stage: Stage) -> bool {
        self.stages.unload(stage)
    }

    /// Whether `stage` holds weights. Reports `true` while a run holds the cell,
    /// and always `true` for `Stage::Translation` under a remote provider, whose
    /// weights are not ours to hold.
    pub(crate) fn loaded(&self, stage: Stage) -> bool {
        self.stages.loaded(stage)
    }

    /// After the last explicit unload: hand torch's cached segments back, under
    /// the same flag the eviction sweep honours. See
    /// `Residency::empty_cache_if_configured` for the whole story.
    pub(crate) fn release_cached_memory(&self) -> bool {
        self.residency.empty_cache_if_configured()
    }

    pub(crate) async fn run(&self, job: StageJob) -> StageCompletion {
        let started = Instant::now();
        let page = job.input.page();
        let model = self.stages.model(job.stage).to_owned();
        let mut waited = Duration::ZERO;
        let outcome = self.run_with_recovery(&job, &model, &mut waited).await;
        StageCompletion {
            page,
            stage: job.stage,
            model,
            elapsed: started.elapsed(),
            waited,
            outcome,
        }
    }

    /// `waited` accumulates time this job spent queued for the accelerator lane
    /// rather than doing its own work.
    ///
    /// It is an out-parameter rather than part of the return type because every
    /// early exit above -- stopped, no work -- must still report a wait, and
    /// three of the four are `return`s that would each have to remember to carry
    /// it. A `&mut` that is written only where a wait actually happens cannot
    /// forget.
    ///
    /// **Why this exists at all.** `run` stamps `started` before calling this,
    /// and `Residency::enter` blocks on a one-permit `Semaphore`, so a stage that
    /// queues behind another has always billed the queue to itself. That is not
    /// a rounding error: measured over a 205-page test volume, `inpainting`
    /// reported 356 ms median while a `clean_only` run of the same pages with no
    /// OCR stage in the pipeline reported 187 ms -- and `inpainting - ocr` sat in
    /// a 106-217 ms band while OCR itself ranged from 1.0 s to 5.9 s, which is
    /// the signature of one number tracking the other. Inpainting is the
    /// *smallest* device stage, and every structural proposal that has been
    /// sized against its reported cost was sized against roughly 2.4x the truth.
    ///
    /// Both waits are counted, and the second is easy to miss: the recovery path
    /// re-enters the lane through `Residency::recover` after an out-of-memory
    /// retry, and that is queueing too.
    async fn run_with_recovery(
        &self,
        job: &StageJob,
        model: &str,
        waited: &mut Duration,
    ) -> std::result::Result<StageOutcome, PipelineError> {
        if job.stop.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        if !self.has_work(job) {
            return Ok(StageOutcome::Skipped);
        }
        let queued = Instant::now();
        let admission = self.residency.enter(job.stage, &self.stages).await;
        *waited += queued.elapsed();
        if job.stop.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        let first = self.run_admitted(job, model, &admission).await;
        let failure = match first {
            Ok(outcome) => {
                self.residency.touch(job.stage, &self.stages);
                return Ok(outcome);
            }
            Err(failure) if is_out_of_memory(&failure.error) && !job.stop.stopped() => {
                self.residency.penalize(job.stage);
                failure
            }
            Err(failure) => return Err(self.stage_error(job.stage, model, failure)),
        };

        drop(admission);
        tracing::warn!(stage = %job.stage, page = %job.input.page(), error = %failure.error, "retrying stage after memory pressure");
        let requeued = Instant::now();
        let recovery = self.residency.recover(job.stage, &self.stages).await;
        *waited += requeued.elapsed();
        if job.stop.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        match self.run_admitted(job, model, &recovery).await {
            Ok(outcome) => {
                self.residency.touch(job.stage, &self.stages);
                Ok(outcome)
            }
            Err(failure) => Err(self.stage_error(job.stage, model, failure)),
        }
    }

    /// Whether this job is worth admitting at all, asked before the accelerator
    /// lane, before residency and before any weights move.
    ///
    /// The obvious home for this is beside the `stages.load` call in
    /// `load_and_process` that it exists to skip. That is the wrong place, for
    /// two reasons that both bite before the load is ever reached.
    ///
    /// `Residency::enter` runs first, and on a card under pressure `unload_idle`
    /// drops *every other* stage to make room for the one being admitted. A gate
    /// below it would still evict the detector, the OCR model and the inpainter
    /// on behalf of a translation stage that then does nothing -- trading one
    /// wasted load for three.
    ///
    /// Worse, `run_admitted` wraps `load_and_process` in a VRAM measurement and
    /// feeds the result to `Residency::observe`. A stage that returned straight
    /// away would be measured moving no memory, and `observe` would write that
    /// down as its profile: `peak_bytes` at the reservation floor rather than
    /// the ~19 GiB the local LLM really needs. Every later page would then be
    /// admitted with every other model left resident, which is how this project
    /// gets a native `abort()` with no Rust error. It is not a corner case
    /// either -- the first slice of the measured webtoon chapter has no text, so
    /// the empty page would be the *first* profile learned, during the profiling
    /// pass that exists precisely to measure the stage alone.
    ///
    /// A processor that cannot answer says `true`, and so does an error: the
    /// same walk runs again inside `process` moments later and reports itself
    /// there with the stage's own context, so nothing is swallowed.
    ///
    /// One thing this does skip: `Residency::touch`, which a no-work stage used
    /// to reach on its way out. It feeds only `unload_idle`'s `sort_by_key`, and
    /// that sort has no accumulator and no early exit -- every candidate is
    /// unloaded whatever the order -- so the recency it maintains decides
    /// nothing today.
    fn has_work(&self, job: &StageJob) -> bool {
        self.stages.has_work(job.stage, &job.input).unwrap_or(true)
    }

    async fn run_admitted(
        &self,
        job: &StageJob,
        model: &str,
        admission: &Admission<'_>,
    ) -> std::result::Result<StageOutcome, AttemptFailure> {
        if !admission.tracked_memory() {
            return self.load_and_process(job, model).await;
        }
        let (outcome, measurement) = self
            .resources
            .measure(self.load_and_process(job, model), admission.profiling())
            .await;
        self.residency
            .observe(job.stage, admission.was_loaded(), measurement);
        outcome
    }

    async fn load_and_process(
        &self,
        job: &StageJob,
        model: &str,
    ) -> std::result::Result<StageOutcome, AttemptFailure> {
        progress::emit(
            job.progress.as_ref(),
            Progress::Loading {
                page: job.input.page(),
                stage: job.stage,
                model: model.to_owned(),
            },
        );
        self.stages
            .load(job.stage)
            .await
            .map_err(|error| AttemptFailure {
                kind: ErrorKind::ModelLoad,
                error,
            })?;
        if job.stop.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        progress::emit(
            job.progress.as_ref(),
            Progress::Running {
                page: job.input.page(),
                stage: job.stage,
                model: model.to_owned(),
            },
        );
        self.stages
            .process(job.stage, job.input.clone())
            .await
            .map(|patch| {
                if patch.is_empty() {
                    StageOutcome::Skipped
                } else {
                    StageOutcome::Patch(patch)
                }
            })
            .map_err(|error| AttemptFailure {
                kind: ErrorKind::Processing,
                error,
            })
    }

    fn stage_error(&self, stage: Stage, model: &str, failure: AttemptFailure) -> PipelineError {
        self.stages.unload(stage);
        let message = match failure.kind {
            ErrorKind::ModelLoad => format!("failed to load {model}"),
            _ => format!("{model} failed"),
        };
        PipelineError::new(failure.kind, Some(stage), failure.error.context(message))
    }
}

struct AttemptFailure {
    kind: ErrorKind,
    error: anyhow::Error,
}

pub(crate) struct StageJob {
    stage: Stage,
    input: StageInput,
    stop: StopToken,
    progress: Option<ProgressSink>,
}

impl StageJob {
    pub(crate) fn new(
        stage: Stage,
        input: StageInput,
        stop: StopToken,
        progress: Option<ProgressSink>,
    ) -> Self {
        Self {
            stage,
            input,
            stop,
            progress,
        }
    }
}

pub(crate) enum StageOutcome {
    Patch(Patch),
    Skipped,
    Stopped,
}

pub(crate) struct StageCompletion {
    pub(crate) page: EntityId,
    pub(crate) stage: Stage,
    pub(crate) model: String,
    pub(crate) elapsed: Duration,
    /// How much of `elapsed` was spent queued for the accelerator lane rather
    /// than doing this stage's own work. Always `<= elapsed`.
    ///
    /// Reported beside `elapsed` rather than subtracted from it, deliberately.
    /// `elapsed` is what the page actually waited for this stage and is the
    /// number a latency budget must keep using; `waited` is what another stage
    /// charged it. Subtracting would silently redefine a field every existing
    /// run artefact already contains, and this project cannot compare a new run
    /// against an old one if a column changes meaning underneath it.
    pub(crate) waited: Duration,
    pub(crate) outcome: std::result::Result<StageOutcome, PipelineError>,
}

/// These exercise the gate's *decision* rather than a run, deliberately. Asking
/// whether a stage with work would be loaded by running it would load it, and
/// the stage under test is the one holding a 16.5 GiB local LLM. `has_work` is
/// the whole of the branch -- `run_with_recovery` does nothing with the answer
/// but return `Skipped` or carry on -- so testing it tests the gate.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use koharu_scene::{At, EntityId, PageDraft, Session, Snapshot, TextLayout, TextLayoutKind};

    use crate::{
        ImageCache, PipelineConfig, ProcessorConfig, Stage, StopToken, resources::ResourceMonitor,
        stages::StageInput,
    };

    use super::{StageJob, StageRunner};

    fn page_with_text(text: Option<&str>) -> (Snapshot, EntityId) {
        let mut session = Session::memory().unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        if let Some(text) = text {
            let content = edit.add_text_content(page, At::End).unwrap();
            edit.set(
                content,
                &koharu_scene::SourceText {
                    text: koharu_scene::Authored::user(text.to_owned()),
                    language: None,
                },
            )
            .unwrap();
            edit.add_text_layer(
                page,
                At::End,
                content,
                &TextLayout {
                    origin: koharu_scene::Origin::User,
                    kind: TextLayoutKind::Paragraph,
                },
            )
            .unwrap();
        }
        session.commit(edit.finish().unwrap()).unwrap();
        (session.snapshot(), page)
    }

    fn job(stage: Stage, text: Option<&str>) -> StageJob {
        let (scene, page) = page_with_text(text);
        StageJob::new(
            stage,
            StageInput::new(
                scene,
                page,
                None,
                None,
                Arc::new(ImageCache::default()),
                None,
                Arc::from([] as [koharu_translator::TranslationContext; 0]),
                Arc::from([] as [koharu_translator::TranslationContext; 0]),
                // No re-roll: the config's own seed, like every non-retry caller.
                None,
                // No reader edits either arm.
                Arc::from([] as [crate::CallerRegion; 0]),
                Arc::from([] as [crate::CallerRegion; 0]),
                // Not a joined page: an ordinary slice, so the size guard applies.
                false,
                Arc::from([] as [f64; 0]),
                None,
            ),
            StopToken::default(),
            None,
        )
    }

    /// Never loads anything: `StageRunner::new` builds empty model cells and
    /// allocates no device memory, which is the same property `Pipeline::reload`
    /// relies on.
    fn runner(skip_empty_stages: bool) -> StageRunner {
        let device = koharu_ml::Device::cpu();
        let config = PipelineConfig {
            processor: ProcessorConfig {
                skip_empty_stages,
                ..ProcessorConfig::default()
            },
            ..PipelineConfig::default()
        };
        let translator = koharu_translator::Translator::from_config(
            device.clone(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        StageRunner::new(&config, translator, &device, ResourceMonitor::new(&device)).unwrap()
    }

    #[test]
    fn a_stage_with_nothing_to_do_is_never_loaded() {
        assert!(!runner(true).has_work(&job(Stage::Translation, None)));
    }

    #[test]
    fn a_stage_with_something_to_do_is_loaded() {
        assert!(runner(true).has_work(&job(Stage::Translation, Some("\u{3042}"))));
    }

    /// The default trait implementation is what keeps this change to one stage.
    /// Detection, OCR and inpainting answer `true` on the same bare page --
    /// detection especially, since it is the stage that would *create* the text
    /// this page has none of.
    #[test]
    fn every_stage_that_has_not_opted_in_behaves_exactly_as_before() {
        let runner = runner(true);
        for stage in [Stage::Detection, Stage::Ocr, Stage::Inpainting] {
            assert!(runner.has_work(&job(stage, None)), "{stage}");
        }
    }

    /// The other arm of the A/B. Off, the gate is inert: the same bare page that
    /// answers `false` above answers `true` here, and the runner is back to
    /// loading first and discovering the page is empty afterwards.
    #[test]
    fn the_flag_off_restores_the_load_then_ask_order() {
        let runner = runner(false);
        for stage in Stage::ALL {
            assert!(runner.has_work(&job(stage, None)), "{stage}");
        }
    }
}
