use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use specta::Type;

use koharu_translator::TranslationContext;

use crate::{ProgressSink, Scope, Stage};
use koharu_scene::EntityId;

#[derive(Clone, Debug)]
pub struct InpaintingMask {
    pub page: EntityId,
    pub png: Arc<[u8]>,
}

/// One caller-stated detection box, in source-image pixel space -- the same
/// space `RegionOut` reports boxes in, so the extension's editor round-trips
/// without a transform. Width and height, not corners: the wire shape is the
/// reader-facing one, and the detection stage converts to the network's
/// corner convention at the one place it constructs a detection from this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallerRegion {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    #[default]
    Full,
    Through {
        stage: Stage,
    },
    Only {
        stage: Stage,
    },
    Stages {
        stages: Vec<Stage>,
    },
}

impl Operation {
    pub(crate) fn stages(&self) -> Result<Vec<Stage>> {
        let stages = match self {
            Self::Full => Stage::ALL.to_vec(),
            Self::Through {
                stage: Stage::Detection,
            } => vec![Stage::Detection],
            Self::Through { stage: Stage::Ocr } => vec![Stage::Detection, Stage::Ocr],
            Self::Through {
                stage: Stage::Translation,
            } => {
                vec![Stage::Detection, Stage::Ocr, Stage::Translation]
            }
            Self::Through {
                stage: Stage::Inpainting,
            } => vec![Stage::Detection, Stage::Inpainting],
            Self::Only { stage } => vec![*stage],
            Self::Stages { stages } => Stage::ALL
                .into_iter()
                .filter(|stage| stages.contains(stage))
                .collect(),
        };
        if stages.is_empty() {
            bail!("at least one pipeline stage must be selected");
        }
        Ok(stages)
    }
}

#[derive(Clone)]
pub struct Request {
    pub operation: Operation,
    pub scope: Scope,
    pub stop: StopToken,
    pub progress: Option<ProgressSink>,
    pub inpainting_mask: Option<InpaintingMask>,
    /// Already-translated source/target pairs to show the model as precedent.
    ///
    /// Belongs on the **request**, not on `TranslationConfig`. `PipelineConfig`
    /// is what a caller compares to decide whether to `reload`, and a whole
    /// pipeline reload discards the learned per-stage residency profiles -- so
    /// context that changes every page would make every page a cold start.
    /// Carried here it changes one prompt and nothing else.
    ///
    /// `Arc<[_]>` because `Request` is cloned per stage job and this is the only
    /// unbounded field on it.
    pub context: Arc<[TranslationContext]>,
    /// Pinned per-series term renderings, law rather than precedent -- see
    /// `TranslationRequest::glossary` for the distinction from `context`.
    ///
    /// On the **request** for the reason `context` above gives: a per-series
    /// store varies it between runs, and on the config that would be a reload.
    pub glossary: Arc<[TranslationContext]>,

    /// This page was ASSEMBLED by the caller so that its text does not continue
    /// past its own edges.
    ///
    /// Set by the extension's webtoon seam, which cuts one image from a run of
    /// consecutive slices precisely because a vertical name crossed the cuts
    /// between them. The run grows until a slice whose text stops inside it, so
    /// the joined image holds the whole name by construction.
    ///
    /// **It exists to disarm `cross_slice_fragment` and nothing else.** That
    /// guard refuses a region reaching both edges of its page, because on an
    /// ORDINARY slice such a region is a fragment of something taller and the
    /// translator fabricates from it. On a joined page the identical shape is
    /// the OPPOSITE situation: the text reaches both edges because the crop was
    /// cut to hold exactly it. Measured on a test webtoon, the joined column is 1693px
    /// of a 1716px seam -- 0.987, past the 0.95 threshold -- so the guard
    /// refused the very join that removed the fragmentation. Padding the crop
    /// does not help: the detector's box grows with it (1784 of 1816).
    ///
    /// **Belongs on the request, not on `PipelineConfig`,** for the reason
    /// `context` above gives: a config that changes per page makes every page a
    /// cold start, and a seam is already one extra full pipeline run per
    /// boundary. `birelate-server` sends a seam's model fields byte-identically
    /// so `needs_reload` stays false; a flag on the config would undo that.
    pub joined_page: bool,
    /// Where the cuts sit inside a joined page, in composite pixels from its
    /// top. Empty means the caller did not say -- as every older caller does
    /// -- and rules that need a cut's position simply do not fire. Read by the
    /// spot rescue's composite gate: a mint that does not SPAN a cut is a
    /// single slice's own business, and without this the two bands adjacent to
    /// one cut column each mint their own fragment and the paint-backs stomp
    /// each other.
    pub joined_boundaries: Vec<f64>,
    /// Boxes the READER drew over text the detector missed.
    /// Admitted into the settled detection list as text regions -- law, not
    /// candidates: NMS may not evict them, and a box holding no ink is the
    /// one refusal they keep (`ink_within`'s floors, which is also what
    /// spares the artwork under a misdrawn rectangle). Empty for every
    /// caller but the extension's box editor.
    ///
    /// On the **request** for `context`'s reason above: the boxes change per
    /// page per click, and on the config every edit would be a reload.
    pub added_regions: Vec<CallerRegion>,
    /// Rectangles the READER marked for deletion: every detection whose box
    /// CENTER lies inside one is dropped after the settle pass -- late
    /// enough to catch a synthesised textless-bubble read (a known
    /// false-positive class) and before the erase masks are written, which
    /// is what genuinely un-erases the artwork underneath. Same carriage as
    /// `added_regions`.
    pub removed_regions: Vec<CallerRegion>,
    /// The translator's sampler seed for THIS run. `None` -- every caller
    /// before the retry button -- leaves `TranslationConfig::generation`
    /// untouched, whose own `seed` the shipping server never sets, so absent
    /// keeps the fixed upstream constant and the byte-identical replay every
    /// measurement arm depends on.
    ///
    /// On the **request** for the reason `context` above gives, and doubly so
    /// here: the whole point of a per-request seed is a caller re-rolling ONE
    /// page, and a seed on `PipelineConfig` would turn every re-roll into a
    /// full pipeline reload. Read only by the translation stage.
    pub translation_seed: Option<u32>,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            operation: Operation::Full,
            scope: Scope::Project,
            stop: StopToken::default(),
            progress: None,
            inpainting_mask: None,
            context: Arc::from([] as [TranslationContext; 0]),
            glossary: Arc::from([] as [TranslationContext; 0]),
            // An ordinary page, whose text may well run off an edge.
            joined_page: false,
            joined_boundaries: Vec::new(),
            // No reader edits: the detector's own answer stands whole.
            added_regions: Vec::new(),
            removed_regions: Vec::new(),
            // The fixed upstream constant -- identical requests stay identical.
            translation_seed: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct StopToken(Arc<AtomicBool>);

impl StopToken {
    pub fn stop(&self) {
        self.0.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn stopped(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}
