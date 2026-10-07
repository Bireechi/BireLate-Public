use std::collections::{BTreeMap, BTreeSet};

use koharu_scene::EntityId;

use crate::Stage;

#[derive(Clone, Copy, Eq, PartialEq)]
enum WorkState {
    Pending,
    Running,
    Finished,
}

struct StageWork {
    stage: Stage,
    state: WorkState,
}

struct PageWork {
    page: EntityId,
    stages: Vec<StageWork>,
}

impl PageWork {
    fn started(&self) -> bool {
        self.stages
            .iter()
            .any(|work| work.state != WorkState::Pending)
    }

    fn finished(&self) -> bool {
        self.stages
            .iter()
            .all(|work| work.state == WorkState::Finished)
    }

    /// Whether this stage may start, given what this page's own stage list holds.
    ///
    /// **An absent prerequisite falls through to ITS prerequisite rather than
    /// counting as satisfied, and getting that wrong silently disabled the
    /// eraser.** This used to be a single lookup that treated "not in the list"
    /// as ready. That was harmless while `Inpainting` depended on `Detection` --
    /// which is never absent -- but then it was made to depend on `Ocr`, and
    /// `clean_only` is `[Detection, Inpainting]` with no OCR stage at all. So
    /// inpainting became ready at once and was dispatched in the same round as
    /// detection, reading a snapshot taken before the mask existed. It ran, was
    /// admitted, was profiled, reported no error, and erased nothing:
    /// `mask_px=0`. On one test page `clean_only` went from erasing 10.02% to
    /// 0.0009%.
    ///
    /// Walking the chain gives the right answer for both shapes: with OCR present
    /// inpainting waits for OCR, and with OCR absent it waits for detection, which
    /// is exactly what it did before the OCR edge. The chain is finite and
    /// acyclic, so the loop terminates.
    fn ready(&self, index: usize) -> bool {
        let mut stage = self.stages[index].stage;
        while let Some(prerequisite) = prerequisite(stage) {
            match self.stages.iter().find(|work| work.stage == prerequisite) {
                Some(work) => return work.state == WorkState::Finished,
                None => stage = prerequisite,
            }
        }
        true
    }
}

pub(crate) struct Scheduler {
    pages: Vec<PageWork>,
    page_index: BTreeMap<EntityId, usize>,
    page_window: usize,
    active_pages: usize,
    head: usize,
    total: usize,
}

impl Scheduler {
    pub(crate) fn new(pages: &[EntityId], stages: &[Stage]) -> Self {
        let pages = pages
            .iter()
            .map(|page| PageWork {
                page: *page,
                stages: stages
                    .iter()
                    .map(|stage| StageWork {
                        stage: *stage,
                        state: WorkState::Pending,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let total = pages.len().saturating_mul(stages.len());
        Self {
            page_index: pages
                .iter()
                .enumerate()
                .map(|(index, page)| (page.page, index))
                .collect(),
            pages,
            page_window: stages.len().max(1),
            active_pages: 0,
            head: 0,
            total,
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    pub(crate) fn start_next(
        &mut self,
        busy_stages: &BTreeSet<Stage>,
    ) -> Option<(EntityId, Stage)> {
        for page_index in self.head..self.pages.len() {
            let started = self.pages[page_index].started();
            if !started && self.active_pages >= self.page_window {
                break;
            }
            let stage_index =
                self.pages[page_index]
                    .stages
                    .iter()
                    .enumerate()
                    .find_map(|(index, work)| {
                        (work.state == WorkState::Pending
                            && !busy_stages.contains(&work.stage)
                            && self.pages[page_index].ready(index))
                        .then_some(index)
                    });
            let Some(stage_index) = stage_index else {
                continue;
            };
            if !started {
                self.active_pages += 1;
            }
            let page = &mut self.pages[page_index];
            let work = &mut page.stages[stage_index];
            work.state = WorkState::Running;
            return Some((page.page, work.stage));
        }
        None
    }

    pub(crate) fn complete_stage(&mut self, page: EntityId, stage: Stage) -> bool {
        let Some(&page_index) = self.page_index.get(&page) else {
            return false;
        };
        let page = &mut self.pages[page_index];
        let was_finished = page.finished();
        if let Some(work) = page.stages.iter_mut().find(|work| work.stage == stage) {
            work.state = WorkState::Finished;
        }
        let page_finished = !was_finished && page.finished();
        if page_finished {
            self.active_pages = self.active_pages.saturating_sub(1);
            while self.head < self.pages.len() && self.pages[self.head].finished() {
                self.head += 1;
            }
        }
        page_finished
    }
}

const fn prerequisite(stage: Stage) -> Option<Stage> {
    match stage {
        Stage::Detection => None,
        Stage::Ocr => Some(Stage::Detection),
        /* INPAINTING WAITS FOR OCR, and it did not used to.
         *
         * It depended on Detection alone, which made the two siblings: the
         * observed Ocr -> Inpainting order came from there being one accelerator
         * lane plus `Stage::ALL` happening to list `Ocr` first, not from any
         * rule. That was harmless while the erase mask was final at the end of
         * detection. It is not harmless now that `ocr.rs` WITHDRAWS unreadable
         * regions from that mask -- under the old edge a second lane, or a
         * reordering of `Stage::ALL`, would let the inpainter consume the mask
         * before the withdrawal was applied, and the artwork would be destroyed
         * exactly as it was before the fix, on some machines and not others.
         *
         * The cost is about nothing today: `inpainting.wait_ms` already equalled
         * `ocr.elapsed` to within 0 ms median on 96% of 277 measured pages, i.e.
         * inpainting was queueing behind OCR regardless. This makes that ordering
         * a guarantee rather than a coincidence.
         *
         * A page with no OCR stage at all -- `clean_only` -- relies on `ready`
         * walking PAST an absent prerequisite to the next one down the chain, so
         * inpainting there waits for detection exactly as it did before. That
         * fall-through did not exist when this edge was first added and the
         * eraser silently did nothing; see `PageWork::ready`. */
        Stage::Inpainting => Some(Stage::Ocr),
        Stage::Translation => Some(Stage::Ocr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(count: usize) -> Vec<EntityId> {
        let session = koharu_scene::Session::memory().unwrap();
        let snapshot = session.snapshot();
        let mut pages = Vec::new();
        snapshot
            .patch(|edit| {
                for index in 0..count {
                    pages.push(edit.add_page(
                        koharu_scene::PageDraft::new(index.to_string(), 1.0, 1.0),
                        koharu_scene::At::End,
                    )?);
                }
                Ok(())
            })
            .unwrap();
        pages
    }

    /// **Inpainting no longer starts beside OCR, and that is the point of this
    /// test rather than an incidental detail of it.**
    ///
    /// It used to: `prerequisite` gave both `Some(Detection)`, so the two were
    /// siblings and this test asserted they became ready together. Now OCR
    /// *edits the erase mask* -- it withdraws regions it could not read,
    /// so a detector false positive stops costing artwork -- and a sibling
    /// inpainting stage is free to snapshot the scene before that edit lands.
    /// The old graph made the fix hold only because one accelerator lane and the
    /// order of `Stage::ALL` happened to run OCR first.
    ///
    /// The cost is real but small: page N's inpainting can no longer overlap page
    /// N's OCR. Page N+1's detection still can, which is what the last assertion
    /// pins, and the server sends one page per request so the loss is confined to
    /// batch runs through `run.exe`.
    #[test]
    fn inpainting_waits_for_ocr_while_the_next_page_runs_ahead() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL);
        let mut busy = BTreeSet::new();

        let first = scheduler.start_next(&busy).unwrap();
        assert_eq!(first, (pages[0], Stage::Detection));
        busy.insert(Stage::Detection);
        assert!(scheduler.start_next(&busy).is_none());

        busy.clear();
        assert!(!scheduler.complete_stage(pages[0], Stage::Detection));
        let ocr = scheduler.start_next(&busy).unwrap();
        busy.insert(ocr.1);
        assert_eq!(ocr, (pages[0], Stage::Ocr));

        // The old graph handed back page 0's inpainting here. It must now hand
        // back the NEXT PAGE's detection instead: inpainting is not ready.
        let next_page = scheduler.start_next(&busy).unwrap();
        busy.insert(next_page.1);
        assert_eq!(next_page, (pages[1], Stage::Detection));

        assert!(!scheduler.complete_stage(pages[0], Stage::Ocr));
        busy.remove(&Stage::Ocr);
        // Only once OCR is finished do BOTH of its dependants open up.
        let translation = scheduler.start_next(&busy).unwrap();
        assert_eq!(translation, (pages[0], Stage::Translation));
        busy.insert(translation.1);
        let inpainting = scheduler.start_next(&busy).unwrap();
        assert_eq!(inpainting, (pages[0], Stage::Inpainting));
        assert!(busy.contains(&Stage::Detection));
    }

    #[test]
    fn sliding_window_backpressures_fast_upstream_models() {
        let pages = pages(4);
        let stages = [Stage::Detection, Stage::Ocr, Stage::Inpainting];
        let mut scheduler = Scheduler::new(&pages, &stages);
        let mut busy = BTreeSet::new();

        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Detection))
        );
        assert!(!scheduler.complete_stage(pages[0], Stage::Detection));
        let ocr = scheduler.start_next(&busy).unwrap();
        busy.insert(ocr.1);
        assert_eq!(ocr, (pages[0], Stage::Ocr));

        // Inpainting is NOT startable here any more -- it waits for OCR -- so the
        // window fills with the detections of later pages instead, which is the
        // backpressure this test exists to check and is unaffected by the change.
        for page in &pages[1..3] {
            assert_eq!(scheduler.start_next(&busy), Some((*page, Stage::Detection)));
            assert!(!scheduler.complete_stage(*page, Stage::Detection));
        }
        busy.insert(Stage::Detection);
        assert!(scheduler.start_next(&busy).is_none());

        assert!(!scheduler.complete_stage(pages[0], Stage::Ocr));
        busy.remove(&Stage::Ocr);
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Inpainting))
        );
        assert!(scheduler.complete_stage(pages[0], Stage::Inpainting));
        busy.clear();
        assert_eq!(scheduler.start_next(&busy), Some((pages[1], Stage::Ocr)));
    }

    /// **`clean_only` is detection + inpainting with no OCR stage at all**, and
    /// inpainting has a prerequisite of OCR. A page whose stage list
    /// does not contain the prerequisite must treat it as satisfied, or the
    /// eraser silently never runs -- which is exactly what shipped.
    #[test]
    fn a_stage_set_without_ocr_still_runs_inpainting() {
        let pages = pages(1);
        let stages = [Stage::Detection, Stage::Inpainting];
        let mut scheduler = Scheduler::new(&pages, &stages);
        let mut busy = BTreeSet::new();
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Detection))
        );
        busy.insert(Stage::Detection);

        // THE ASSERTION THIS TEST WAS MISSING, and its absence shipped a silent
        // regression. The first version completed detection here and then checked
        // that inpainting could start -- which it could, so the test passed while
        // `clean_only` erased nothing. What was broken is that inpainting could
        // ALSO start right now, in the same round, against a snapshot taken
        // before detection had written the mask.
        assert!(
            scheduler.start_next(&busy).is_none(),
            "inpainting must not start before detection has written the mask"
        );

        busy.clear();
        assert!(!scheduler.complete_stage(pages[0], Stage::Detection));
        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Inpainting)),
            "clean_only must still reach the inpainter"
        );
    }
}
