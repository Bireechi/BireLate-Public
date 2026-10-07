use std::sync::Arc;

use crate::Stage;
use koharu_scene::EntityId;

#[derive(Clone, Debug)]
pub enum Progress {
    Started {
        pages: Vec<EntityId>,
        stages: Vec<Stage>,
    },
    Loading {
        page: EntityId,
        stage: Stage,
        model: String,
    },
    Running {
        page: EntityId,
        stage: Stage,
        model: String,
    },
    Finished {
        page: EntityId,
        stage: Stage,
        model: String,
        elapsed: std::time::Duration,
        /// The part of `elapsed` this stage spent queued for the accelerator
        /// lane instead of working. See `StageCompletion::waited`.
        waited: std::time::Duration,
    },
    Skipped {
        page: EntityId,
        stage: Stage,
    },
    /// The translation stage finished, but its reply did not fully cover the
    /// page: some text came back in the source language, or generation was cut
    /// off, or both.
    ///
    /// Reported rather than acted on: a partial page is still worth rendering,
    /// and only the caller knows whether its user would rather see it or see an
    /// error. What the caller cannot do is *notice* on its own -- the affected
    /// entities carry a `Translation` like any other, holding the source text
    /// verbatim, so the finished scene looks exactly as it would if the OCR
    /// stage had never read those bubbles.
    ///
    /// `entities` empty with `truncated` set is a real and common shape, not a
    /// contradiction: a reply cut off inside the last segment's text is
    /// repaired into a complete entry, so every id is answered and the only
    /// damage is one bubble ending mid-word. A caller that gates on `entities`
    /// alone will call that page clean.
    ///
    /// `entities` empty with `truncated` clear is a third shape, and it is the
    /// reason this event no longer implies a token cap: a provider free to
    /// answer more entries than were asked for can duplicate an id and still
    /// cover every one, which damages nothing on the page but says the id
    /// discipline slipped. Report the fields, never the variant's name.
    Untranslated {
        page: EntityId,
        /// The text content entities still holding their source text, in the
        /// order the stage submitted them. Empty when the reply answered every
        /// id -- see `truncated`.
        entities: Vec<EntityId>,
        /// How many segments the stage submitted in total, so a caller can say
        /// "3 of 41" without walking the scene. `entities.len() == segments` is
        /// a page that came back wholly untranslated.
        segments: usize,
        /// Generation stopped on the model's token cap, which is the usual
        /// cause on a dense page. `false` on every remote provider means only
        /// that the finish reason was not available.
        truncated: bool,
        /// Reply entries the translator dropped because their id had already
        /// been answered.
        ///
        /// This is the only thing that tells a caller *why* `entities` is not
        /// empty. That list is derived from a filled-in mask, so a duplicate
        /// shows up in it only as its victim -- the id that consequently went
        /// unanswered -- and reads exactly like a reply that stopped early.
        /// "The model never answered these" wants a bigger token budget;
        /// "the model answered these twice" wants nothing of the sort.
        duplicate_ids: usize,
        /// Reply entries naming an id no submitted segment has, likewise
        /// dropped. Kept apart from `duplicate_ids` because under the local
        /// provider the id is schema-constrained and this should be impossible,
        /// so a non-zero count accuses the grammar rather than the model.
        out_of_range_ids: usize,
        /// Segments whose first reply arrived visibly cut mid-sentence
        /// -- the cut repair's input population, counted even when
        /// the repair arm is off.
        cut_found: usize,
        /// Segments that SHIP visibly cut after any repair -- the final text
        /// recounted, so a failed retry is included. `cut_found > 0` with
        /// `still_cut == 0` is a page that was cut and repaired; a non-zero
        /// `still_cut` is a bubble ending mid-word right now.
        still_cut: usize,
    },
    /// The detection stage saw text against a page edge that it deliberately did
    /// **not** admit as a region, because the detector scored it under the
    /// checkpoint's own `text` floor.
    ///
    /// Reported rather than acted on, for exactly the reason `Untranslated`
    /// above is, and the "acted on" here would be actively harmful. Admitting a
    /// sub-floor box as a region does not merely letter a doubtful detection:
    /// `stages/detection.rs` `mask_includes` builds the ERASE MASK from the raw
    /// detections, before OCR has read anything, so a fragment admitted here
    /// would have its artwork stripped on every ordinary page read while nothing
    /// ever letters it. The pipeline refuses a cross-slice fragment in two places
    /// on purpose so that pairing cannot come apart, and this event exists so a
    /// low-confidence box can be *mentioned* without joining it.
    ///
    /// So the only caller this is for is one **assembling a page out of slices**.
    /// A display column running down a webtoon can be clearly present in a slice
    /// and still score 0.2363 against a 0.25 floor -- measured, on a test
    /// webtoon slice -- and a caller deciding which slices to join needs to know the
    /// ink is there. What it must not do is treat a hint as a region.
    ///
    /// Emitted only when the page has at least one; a run that found none is
    /// silent, like `Untranslated`. Note the corollary that silence also covers
    /// "detection did not run on this page at all", which is the ordinary shape
    /// of a re-render.
    EdgeHints {
        page: EntityId,
        /// Every sub-floor text box touching exactly one of the page's edge
        /// bands, in detection order. Never empty -- see the variant's doc.
        hints: Vec<EdgeHint>,
    },
}

/// Which single edge of the page a box is up against.
///
/// Deliberately not "both": a box touching both bands is already spanning its
/// slice and there is nothing for a joiner to decide about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageEdge {
    Top,
    Bottom,
}

impl PageEdge {
    /// The wire spelling, fixed. `birelate-server` puts this straight on
    /// `TranslateJson::edge_hints[].edge` and `extension/seam.js` matches on it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Top => "top",
            Self::Bottom => "bottom",
        }
    }
}

/// One sub-floor text box against a page edge. See [`Progress::EdgeHints`].
///
/// `(x, y, width, height)` rather than the detector's
/// `[left, top, right, bottom]`, because that is the shape every region already
/// crosses the wire in and a hint is read beside them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeHint {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub edge: PageEdge,
    /// What the detector actually scored this box, which is the whole reason it is
    /// a hint and not a region: it is under the run's `text_floor`.
    ///
    /// Carried because without it a refused box is **unfalsifiable**. A region
    /// reports `detection_confidence`; without this field a hint would report
    /// nothing, so a sub-floor score that a join decision rests on could not be
    /// re-derived from the server's own output.
    ///
    /// Not clamped, rounded or thresholded on the way out. A caller comparing it
    /// against `EDGE_HINT_MIN_SCORE` or the text floor needs the number the
    /// detector produced, not one this struct decided was tidy. (Named in prose
    /// rather than linked: that constant is private to the detection stage, and an
    /// intra-doc link to it is a rustdoc warning.)
    pub score: f32,
    /// Whether this box's own ink was walked and found to cross the slice END TO
    /// END, in which case `(x, y, width, height)` above is the **grown** box and
    /// reaches both edge bands.
    ///
    /// # This flag exists so the browser does NOT have to guess from geometry
    ///
    /// `seam.js` drops a hint touching both bands, on purpose: `seamSpansSlice`
    /// tests object identity across the two edge lists, so "one object in both"
    /// is what declares a slice crossed end to end and decides how many slices a
    /// run covers. A sub-floor fragment must not reach that lever by accident.
    ///
    /// That rule is **kept**. What this adds is a narrower, server-attested door:
    /// `false` behaves exactly as before, and only `true` — meaning the ink walk
    /// actually followed this box's ink from the band it touches to the opposite
    /// one — admits the box to both lists. The browser cannot make that judgement
    /// itself; it never sees a pixel.
    ///
    /// The walk is [`repaired_column`]'s, called on the same detection and reused
    /// rather than re-derived, so a sub-floor box is grown by exactly the rule an
    /// above-floor one is — and the guards that keep the whole mechanism off manga
    /// (touches exactly ONE band, column-shaped, ink must really reach) are the
    /// same guards, not a second copy that can drift.
    ///
    /// [`repaired_column`]: crate::stages::detection
    pub spans: bool,
}

pub type ProgressSink = Arc<dyn Fn(Progress) + Send + Sync>;

pub(crate) fn emit(sink: Option<&ProgressSink>, progress: Progress) {
    if let Some(sink) = sink
        && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sink(progress))).is_err()
    {
        tracing::warn!("pipeline progress callback panicked");
    }
}
