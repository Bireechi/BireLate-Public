use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context as _, Result, ensure};
use arc_swap::ArcSwap;
use futures::{StreamExt as _, stream::FuturesUnordered};
use koharu_config::Config;
use koharu_scene::{EntityId, Snapshot};

use crate::{
    Committer, ErrorKind, PipelineConfig, PipelineError, Progress, ProgressSink, Report, Request,
    ResourceSnapshot, RunStatus, Stage, StageOutput, StopToken,
    images::ImageCache,
    progress,
    resources::ResourceMonitor,
    scheduler::Scheduler,
    scope::NormalizedScope,
    stage_runner::{StageCompletion, StageJob, StageOutcome, StageRunner},
    stages::StageInput,
};

#[derive(Clone)]
pub struct Pipeline {
    current: Arc<ArcSwap<StageRunner>>,
    resources: Arc<ResourceMonitor>,
    execution: Arc<tokio::sync::Mutex<()>>,
    /// Retained so `reload` can rebuild a `StageRunner` without going through
    /// the watcher task, which publishes nothing a caller can await.
    translator: koharu_translator::Translator,
    device: koharu_ml::Device,
}

impl Pipeline {
    pub fn load(device: koharu_ml::Device) -> Result<Self> {
        Self::from_config(
            PipelineConfig::load()?,
            koharu_translator::ProvidersConfig::load()?,
            device,
        )
    }

    pub fn from_config(
        config: Config<PipelineConfig>,
        providers: Config<koharu_translator::ProvidersConfig>,
        device: koharu_ml::Device,
    ) -> Result<Self> {
        let translator = koharu_translator::Translator::from_config(device.clone(), providers)?;
        let resources = ResourceMonitor::new(&device);
        let runner = {
            let value = config.read()?;
            StageRunner::new(&value, translator.clone(), &device, resources.clone())?
        };
        let current = Arc::new(ArcSwap::from_pointee(runner));
        let watched = current.clone();
        let watched_resources = resources.clone();
        let owned_translator = translator.clone();
        let owned_device = device.clone();
        let _watcher = tokio::runtime::Handle::try_current()
            .context("pipeline requires a Tokio runtime")?
            .spawn(async move {
                let mut changes = config.subscribe();
                while changes.changed().await.is_ok() {
                    let runner = config.read().and_then(|value| {
                        StageRunner::new(
                            &value,
                            translator.clone(),
                            &device,
                            watched_resources.clone(),
                        )
                    });
                    match runner {
                        Ok(runner) => watched.store(Arc::new(runner)),
                        Err(error) => tracing::error!(%error, "failed to reload pipeline"),
                    }
                }
            });
        Ok(Self {
            current,
            resources,
            execution: Arc::new(tokio::sync::Mutex::new(())),
            translator: owned_translator,
            device: owned_device,
        })
    }

    /// Installs `config` synchronously, replacing the active stage runner.
    ///
    /// Writing the watched [`Config`] instead would leave the caller with no way
    /// to tell when the new runner became visible to [`Pipeline::execute`], and
    /// neither the target language nor the LLM name is reported by [`Progress`],
    /// so a late swap there is undetectable.
    ///
    /// `StageRunner::new` allocates no device memory -- every processor starts
    /// with an empty model cell and loads lazily -- and `ArcSwap::store` drops
    /// the previous runner, so the old weights are released before the new ones
    /// are loaded. Call it only when no execution is in flight.
    pub fn reload(&self, config: &PipelineConfig) -> Result<()> {
        let runner = StageRunner::new(
            config,
            self.translator.clone(),
            &self.device,
            self.resources.clone(),
        )?;
        self.current.store(Arc::new(runner));
        Ok(())
    }

    /// Loads every stage's weights, so the first real page does not pay for it.
    ///
    /// Deliberately outside residency admission control: admission exists to
    /// decide which weights to evict so that a stage can *run*, and nothing is
    /// running here. It also learns nothing, because a profile is measured
    /// around a run rather than around a load, so a warmup neither creates a
    /// profile nor spends one. Under a remote translation provider the
    /// translation stage is a no-op -- those weights live in another process.
    ///
    /// Every stage is resident at once when this returns, which is more than a
    /// single page needs at any one instant. Call it only when the caller has
    /// satisfied itself there is room, and only when no execution is in flight.
    pub async fn load_models(&self) -> Result<()> {
        let runner = self.current.load_full();
        for stage in Stage::ALL {
            runner
                .load(stage)
                .await
                .with_context(|| format!("failed to load the {stage} model"))?;
        }
        Ok(())
    }

    /// Drops every stage's weights, keeping the stage runner -- and with it the
    /// residency profiles -- alive. Reports whether anything was actually freed.
    ///
    /// [`Pipeline::reload`] frees the weights too, but it is the wrong tool for
    /// an idle budget: it builds a fresh stage runner, whose fresh residency has
    /// none of the per-stage VRAM profiles learned so far. An unprofiled stage
    /// is then admitted by evicting *every* other model, including the one being
    /// admitted, so that it can be measured alone. The first page after every
    /// nap would pay a cold start's price, permanently.
    ///
    /// `false` means either that nothing was loaded or that a model cell was
    /// locked: the unload uses `try_lock` and gives up rather than wait, so an
    /// unload racing a run silently does nothing. [`Pipeline::execute`] takes a
    /// private lock of its own, so a caller outside this crate needs a mutex of
    /// its own around the two; while it is held the second case cannot arise.
    ///
    /// When something was freed, torch's cached segments are handed back under
    /// the same flag the eviction sweep honours (dropping a model returns its
    /// tensors to torch's caching allocator, not
    /// to the driver, so without this the explicit path left ~1.23 GiB parked
    /// on the card that `/unload`'s own log then reported as "freed"). Safe
    /// even against the silent-race case above: the cache-empty only releases
    /// segments with no live block.
    #[must_use]
    pub fn unload_models(&self) -> bool {
        let runner = self.current.load();
        let mut freed = false;
        for stage in Stage::ALL {
            if runner.unload(stage) {
                tracing::debug!(%stage, "unloaded stage model");
                freed = true;
            }
        }
        if freed {
            runner.release_cached_memory();
        }
        freed
    }

    /// Whether `stage` holds weights right now.
    ///
    /// Two caveats for anyone folding these into a "the models are loaded" flag.
    /// A cell held by a run in flight reports `true` rather than blocking. And
    /// [`Stage::Translation`] reports `true` under every non-local provider,
    /// which holds no weights of ours at all, so a stone-cold process would look
    /// warm if that stage were counted.
    #[must_use]
    pub fn model_loaded(&self, stage: Stage) -> bool {
        self.current.load().loaded(stage)
    }

    /// The translator behind the translation stage.
    ///
    /// [`Pipeline::reload`] deliberately reuses it, so a config swap keeps any
    /// local LLM weights resident. Exposed so a caller enforcing an idle budget
    /// can drop them on its own schedule. [`Pipeline::unload_models`] already
    /// reaches them through [`Stage::Translation`], so unloading here straight
    /// afterwards reports `false`: there is nothing left to take.
    #[must_use]
    pub fn translator(&self) -> &koharu_translator::Translator {
        &self.translator
    }

    pub fn subscribe_resources(&self) -> tokio::sync::watch::Receiver<ResourceSnapshot> {
        self.resources.start();
        self.resources.subscribe()
    }

    pub async fn execute(
        &self,
        snapshot: Snapshot,
        request: Request,
        committer: &mut dyn Committer,
    ) -> std::result::Result<Report, PipelineError> {
        let _execution = self.execution.lock().await;
        Execution::new(
            self.current.load_full(),
            self.resources.clone(),
            snapshot,
            request,
            committer,
        )?
        .run()
        .await
    }
}

struct Execution<'a> {
    runner: Arc<StageRunner>,
    resources: Arc<ResourceMonitor>,
    committer: &'a mut dyn Committer,
    stop: StopToken,
    progress: Option<ProgressSink>,
    scope: NormalizedScope,
    scheduler: Scheduler,
    scene: Snapshot,
    images: BTreeMap<EntityId, Arc<ImageCache>>,
    busy_stages: BTreeSet<Stage>,
    completed: usize,
    failure: Option<PipelineError>,
    base: koharu_scene::Revision,
    started: Instant,
    inpainting_mask: Option<crate::InpaintingMask>,
    context: Arc<[koharu_translator::TranslationContext]>,
    /// See `Request::glossary`; carried per run beside `context`.
    glossary: Arc<[koharu_translator::TranslationContext]>,
    /// See `Request::joined_page`. Carried per run, never on the config, so a
    /// seam does not force a pipeline reload.
    joined_page: bool,
    /// See `Request::joined_boundaries`; carried per run beside `joined_page`.
    joined_boundaries: std::sync::Arc<[f64]>,
    /// See `Request::added_regions` / `removed_regions`; carried per run so an
    /// edit costs no reload. Read only by the detection stage.
    added_regions: std::sync::Arc<[crate::CallerRegion]>,
    removed_regions: std::sync::Arc<[crate::CallerRegion]>,
    /// See `Request::translation_seed`; carried per run so a re-roll costs no
    /// reload. Read only by the translation stage.
    translation_seed: Option<u32>,
}

impl<'a> Execution<'a> {
    fn new(
        runner: Arc<StageRunner>,
        resources: Arc<ResourceMonitor>,
        snapshot: Snapshot,
        request: Request,
        committer: &'a mut dyn Committer,
    ) -> std::result::Result<Self, PipelineError> {
        let started = Instant::now();
        let base = snapshot.revision();
        let stages = request
            .operation
            .stages()
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?;
        let scope = NormalizedScope::new(&snapshot, &request.scope, &stages)
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?;
        let pages = scope.pages().to_vec();
        if let Some(mask) = request.inpainting_mask.as_ref()
            && (!pages.contains(&mask.page) || !stages.contains(&Stage::Inpainting))
        {
            return Err(PipelineError::new(
                ErrorKind::InvalidInput,
                Some(Stage::Inpainting),
                anyhow::anyhow!("the inpainting mask page is outside the inpainting scope"),
            ));
        }
        progress::emit(
            request.progress.as_ref(),
            Progress::Started {
                pages: pages.clone(),
                stages: stages.clone(),
            },
        );

        Ok(Self {
            runner,
            resources,
            committer,
            stop: request.stop,
            progress: request.progress,
            scope,
            scheduler: Scheduler::new(&pages, &stages),
            scene: snapshot,
            images: BTreeMap::new(),
            busy_stages: BTreeSet::new(),
            completed: 0,
            failure: None,
            base,
            started,
            inpainting_mask: request.inpainting_mask,
            context: request.context,
            glossary: request.glossary,
            joined_page: request.joined_page,
            joined_boundaries: std::sync::Arc::from(request.joined_boundaries.as_slice()),
            added_regions: std::sync::Arc::from(request.added_regions.as_slice()),
            removed_regions: std::sync::Arc::from(request.removed_regions.as_slice()),
            translation_seed: request.translation_seed,
        })
    }

    async fn run(mut self) -> std::result::Result<Report, PipelineError> {
        if self.stopped() {
            return Ok(self.report(RunStatus::Stopped));
        }

        self.resources.start();
        self.resources.wait_for_sample().await;

        let runner = self.runner.clone();
        let mut running = FuturesUnordered::new();
        loop {
            while let Some(job) = self.take_ready_job() {
                running.push(runner.run(job));
            }

            let Some(completion) = running.next().await else {
                break;
            };
            self.busy_stages.remove(&completion.stage);
            if self.stopped() || self.failure.is_some() {
                continue;
            }
            if let Err(error) = self.apply_completion(completion).await {
                self.failure = Some(error);
            }
        }

        self.finalize()
    }

    fn take_ready_job(&mut self) -> Option<StageJob> {
        if self.stopped() || self.failure.is_some() {
            return None;
        }
        let (page, stage) = self.scheduler.start_next(&self.busy_stages)?;
        self.busy_stages.insert(stage);
        let images = self
            .images
            .entry(page)
            .or_insert_with(|| Arc::new(ImageCache::default()))
            .clone();
        Some(StageJob::new(
            stage,
            StageInput::new(
                self.scene.clone(),
                page,
                self.scope.entities(),
                self.scope.region(page),
                images,
                self.inpainting_mask
                    .as_ref()
                    .filter(|mask| stage == Stage::Inpainting && mask.page == page)
                    .cloned(),
                self.context.clone(),
                self.glossary.clone(),
                self.translation_seed,
                self.added_regions.clone(),
                self.removed_regions.clone(),
                self.joined_page,
                self.joined_boundaries.clone(),
                self.progress.clone(),
            ),
            self.stop.clone(),
            self.progress.clone(),
        ))
    }

    async fn apply_completion(
        &mut self,
        completion: StageCompletion,
    ) -> std::result::Result<(), PipelineError> {
        let StageCompletion {
            page,
            stage,
            model,
            elapsed,
            waited,
            outcome,
        } = completion;
        match outcome? {
            StageOutcome::Stopped => {}
            StageOutcome::Skipped => {
                self.mark_complete(page, stage);
                progress::emit(self.progress.as_ref(), Progress::Skipped { page, stage });
            }
            StageOutcome::Patch(patch) => {
                if !self.commit_patch(page, stage, patch).await? {
                    return Ok(());
                }
                self.mark_complete(page, stage);
                progress::emit(
                    self.progress.as_ref(),
                    Progress::Finished {
                        page,
                        stage,
                        model,
                        elapsed,
                        waited,
                    },
                );
            }
        }
        Ok(())
    }

    async fn commit_patch(
        &mut self,
        page: EntityId,
        stage: Stage,
        patch: koharu_scene::Patch,
    ) -> std::result::Result<bool, PipelineError> {
        let patch = patch
            .rebase_on(&self.scene)
            .and_then(|patch| {
                patch.validate_on(&self.scene)?;
                Ok(patch.with_label(format!("Pipeline {stage} for page {page}")))
            })
            .context("failed to rebase stage output onto the latest scene")
            .map_err(|error| PipelineError::new(ErrorKind::InvalidOutput, Some(stage), error))?;
        if self.stopped() {
            return Ok(false);
        }

        let next = self
            .committer
            .commit(StageOutput { page, stage, patch })
            .await
            .with_context(|| format!("failed to commit {stage} output for page {page}"))
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        validate_commit(&self.scene, &next)
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        self.scene = next;
        Ok(true)
    }

    fn mark_complete(&mut self, page: EntityId, stage: Stage) {
        if self.scheduler.complete_stage(page, stage) {
            self.images.remove(&page);
        }
        self.completed += 1;
    }

    fn stopped(&self) -> bool {
        self.stop.stopped()
    }

    fn finalize(mut self) -> std::result::Result<Report, PipelineError> {
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        if !self.stopped() && self.completed != self.scheduler.total() {
            return Err(PipelineError::new(
                ErrorKind::InvalidOutput,
                None,
                anyhow::anyhow!(
                    "pipeline scheduler stopped after {} of {} work items",
                    self.completed,
                    self.scheduler.total()
                ),
            ));
        }
        let status = if self.stopped() {
            RunStatus::Stopped
        } else {
            RunStatus::Completed
        };
        Ok(self.report(status))
    }

    fn report(&self, status: RunStatus) -> Report {
        report(
            status,
            self.base,
            self.scene.revision(),
            self.completed,
            self.scheduler.total(),
            self.started,
        )
    }
}

fn validate_commit(previous: &Snapshot, next: &Snapshot) -> Result<()> {
    ensure!(
        previous.project_id() == next.project_id(),
        "committer returned a snapshot from another project"
    );
    ensure!(
        next.revision() > previous.revision(),
        "committer did not advance the scene revision"
    );
    Ok(())
}

fn report(
    status: RunStatus,
    base: koharu_scene::Revision,
    final_revision: koharu_scene::Revision,
    completed: usize,
    total: usize,
    started: Instant,
) -> Report {
    Report {
        status,
        base,
        final_revision,
        completed,
        total,
        elapsed: started.elapsed(),
    }
}
