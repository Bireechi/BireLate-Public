//! Translation context carried across the pages of one story.
//!
//! A manga chapter is not a set of unrelated pictures. Names, honorifics,
//! register and who is speaking all run across page breaks, and without this
//! every page would be translated in complete isolation: a fresh scene, one LLM
//! call, no memory. Koharu's own `TranslationRequest` has always had a `context`
//! field that `prompt.rs` serialises into every prompt -- it just had no caller,
//! because `with_context` was `#[cfg(test)]`. This fork ungates it and hangs it
//! off the pipeline `Request`; this is what fills it.
//!
//! **The story boundary is declared to the SERVER, never inferred by it.** This
//! file takes whatever `story` id the caller sends and never guesses one: a
//! request may arrive from any client, and a server-side guess would silently
//! merge two readers' chapters.
//!
//! **Where that id comes from is a client question.** The extension derives it
//! from the page URL -- `storyIdFor` (`extension/story.js`) is documented in
//! `background.js` as *"a pure function of the URL"*, is TOTAL (every URL and
//! every non-URL yields an id), and `background.js` attaches it to every
//! translate request. "New story" only bumps a reset counter that changes the
//! derived id; it is not what creates one.
//!
//! Story context is therefore not opt-in for a reader, and a test harness that
//! omits the field is in a configuration no reader is ever in. An extension
//! session ALWAYS carries a story; a bare `curl` never does, and their
//! translations differ on the same binary: a name can come back as an invented
//! transliteration with no window and as a plain English rendering with one.
//!
//! Server-side rather than client-side for one reason that decides it: if the
//! extension carried the context it would have to send it up on every request
//! and receive the new pairs back, which means asking for `format=json` and
//! re-encoding the page, and it would put an unbounded blob in the multipart
//! body. Here it is a string id in, and nothing out.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use koharu_translator::TranslationContext;

/// How many source/target pairs a story carries into the next page's prompt when
/// `--story-pairs` is not given. The flag's own doc comment in `cli.rs` carries
/// what 96 costs, what it buys, and the ceiling.
///
/// This is a **VRAM** number, not a taste one. `context_values` sets
/// `n_ctx = prompt + max_tokens + 1` when unset, and this model's KV costs
/// ~880 KiB per token, so prompt tokens are charged at the same rate as
/// generated ones. A bubble and its English run about 30-40 tokens once the JSON
/// scaffolding is counted.
///
/// It also has to stay well clear of the translation token budget: that scales
/// `max_tokens` to the page and caps it, and the two share `n_ctx`.
///
/// **A window of 24 was measurably too small for a webtoon.** It was sized
/// against a manga page of 10-20 regions, where 24 pairs is a page and a half of
/// dialogue. A webtoon slice carries **0.72** pairs -- measured over 219 slices
/// of a test chapter -- so the same window covered only the last dozen slices
/// and `story_context` saturated at its cap by slice 20 of 219. The consequence
/// was measured, not argued: a term recurring within ~40 slices held one
/// rendering, and the chapter's central artifact, whose occurrences sit 67 and
/// 76 slices apart, came back three different ways.
///
/// **Measured at 96 over the same chapter: the drift is gone.** The artifact's
/// name came back as one rendering on all five occurrences against three at 24,
/// two other recurring terms stayed at one each, OCR was identical on all 219
/// slices, and a blind three-lens panel over the 102 regions whose English moved
/// split 22-21 with 29 ties -- i.e. sentence quality is a wash and the naming is
/// the whole gain. `story_context` reached the new cap on 98 of 219 pages.
///
/// **This does NOT need the reduced sliding-window KV cache.** Measured
/// `n_ctx` peaked at 3840, which is 3.22 GiB of KV with llama.cpp's full
/// sliding-window cache -- comfortably inside
/// the translation stage's allowance, so the window costs no wall time at all.
/// `--swa-full false` would cut that to 1.46 GiB but costs ~14% of every page,
/// which is a bad trade at this size. It becomes the right call only for a
/// window several times larger again.
pub const DEFAULT_PAIRS: usize = 96;

/// How many stories are remembered at once.
///
/// Small on purpose. A reader has one story open; this exists so that pressing
/// "New story" does not have to reach the server to be correct, and so a stale
/// id from a reopened popup cannot resurrect a story the reader ended.
const MAX_STORIES: usize = 8;

/// Longest id accepted. Ids are minted by the popup as a UUID.
const MAX_ID_LEN: usize = 64;

/// The rolling per-story window of already-translated pairs.
///
/// One `Mutex` over the whole map, held only for the length of a clone or a
/// push. The GPU gate already serialises the expensive part of a request, so
/// there is nothing to gain from finer locking and a `RwLock` would only add a
/// second way to get the ordering wrong.
pub struct Stories {
    inner: Mutex<HashMap<String, Vec<TranslationContext>>>,
    /// The window, from `--story-pairs`. Held on the store rather than read at
    /// the call site because the store is built **once**, at startup: a window
    /// consulted anywhere else would be a second copy of the setting, and the
    /// two could disagree for the life of the process. `0` means no window.
    max_pairs: usize,
}

/// `DEFAULT_PAIRS`, for tests and for anything that only needs *a* store.
///
/// The server never takes this path -- `lib.rs` calls `new` with the resolved
/// flag -- and `new` has no default argument precisely so that a configured
/// window cannot be silently replaced by this one.
impl Default for Stories {
    fn default() -> Self {
        Self::with_window(DEFAULT_PAIRS)
    }
}

impl Stories {
    /// `max_pairs` is `--story-pairs`, already validated by `cli::resolve`.
    #[must_use]
    pub fn new(max_pairs: usize) -> Arc<Self> {
        Arc::new(Self::with_window(max_pairs))
    }

    fn with_window(max_pairs: usize) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            max_pairs,
        }
    }

    /// The pairs to show the model for `id`, oldest first.
    ///
    /// An unknown id is not an error: the first page of a story is exactly the
    /// case where there is nothing to carry, and a server restarted mid-chapter
    /// should keep translating rather than refuse.
    #[must_use]
    pub fn context(&self, id: &str) -> Arc<[TranslationContext]> {
        let Ok(stories) = self.inner.lock() else {
            return Arc::from([] as [TranslationContext; 0]);
        };
        stories
            .get(id)
            .map_or_else(|| Arc::from([] as [TranslationContext; 0]), |pairs| {
                Arc::from(pairs.as_slice())
            })
    }

    /// Appends what this page said, and trims to the window.
    ///
    /// Pairs whose source and translation are equal are dropped. That is the
    /// shape an *untranslated* segment takes -- `prompt::translations` seeds its
    /// result from the source and fills in only the ids the model returned, so a
    /// segment the model skipped comes back holding the Japanese verbatim.
    /// Feeding those forward would teach the model that Japanese in, Japanese
    /// out is the expected answer, which is precisely the failure the pipeline's
    /// untranslated-region report exists to make visible.
    pub fn record(&self, id: &str, pairs: impl IntoIterator<Item = (String, String)>) {
        if self.max_pairs == 0 {
            /* `--story-pairs 0` is the no-window arm. Returning here rather than
             * letting the trim below drain everything keeps the map empty, so
             * `MAX_STORIES` eviction never runs and a stale id cannot occupy a
             * slot in a store that can never answer with anything. */
            return;
        }
        let fresh = pairs
            .into_iter()
            .filter(|(source, translation)| {
                let source = source.trim();
                let translation = translation.trim();
                !source.is_empty()
                    && !translation.is_empty()
                    && source != translation
                    && !is_degenerate_repetition(translation)
            })
            .map(|(source, translation)| TranslationContext {
                source,
                translation,
            })
            .collect::<Vec<_>>();
        if fresh.is_empty() {
            return;
        }
        let Ok(mut stories) = self.inner.lock() else {
            return;
        };
        if !stories.contains_key(id) && stories.len() >= MAX_STORIES {
            // Whole stories, not pairs: dropping the least-recently-grown story
            // is the only eviction that cannot silently truncate the one being
            // read right now.
            if let Some(victim) = stories
                .iter()
                .min_by_key(|(_, pairs)| pairs.len())
                .map(|(key, _)| key.clone())
            {
                stories.remove(&victim);
            }
        }
        let entry = stories.entry(id.to_owned()).or_default();
        entry.extend(fresh);
        if entry.len() > self.max_pairs {
            // Keep the newest. Dialogue nearest the current page is the part
            // still carrying an antecedent.
            let excess = entry.len() - self.max_pairs;
            entry.drain(..excess);
        }
    }

    /// Forgets one story, for the popup's "New story" button.
    ///
    /// Returns whether anything was there, so the popup can say "cleared" rather
    /// than guess.
    pub fn clear(&self, id: &str) -> bool {
        self.inner
            .lock()
            .is_ok_and(|mut stories| stories.remove(id).is_some())
    }

    /// How many pairs a story is carrying. For `/status` and for tests.
    #[must_use]
    pub fn len(&self, id: &str) -> usize {
        self.inner
            .lock()
            .map_or(0, |stories| stories.get(id).map_or(0, Vec::len))
    }
}

/// Words a translation must reach before its repetition is judged at all.
///
/// Short repetition is ordinary lettering -- a laugh, a heartbeat, a running
/// sound effect -- and a rule that called those degenerate would strip exactly
/// the register manga is written in. Measured over 952 pairs of twelve words or
/// more, nothing legitimate comes close to the threshold below, so twelve is
/// where judging becomes safe rather than where it becomes interesting.
const REPETITION_MIN_WORDS: usize = 12;

/// Least share of distinct words a long translation must carry.
///
/// **Sits in an empty band two orders of magnitude wide, which is the whole
/// argument for the number.** Measured over the same 952 pairs: the four
/// degenerate ones score 0.002-0.005 and the next lowest legitimate pair scores
/// **0.714**, with nothing whatever in between. 0.20 is not a tuned value, it is
/// the middle of a gap.
const REPETITION_MIN_UNIQUE: f32 = 0.20;

/// Whether a translation is a degenerate repetition loop rather than a sentence.
///
/// **This exists because one such pair poisoned the window for the pages after
/// it, and the poisoning is the expensive half.** One page of a manga test volume
/// came back with a sound effect whose translation is 2,131 characters of one
/// word repeated ~530 times. It cost that page 16.5s of the volume's slowest
/// generation, and `Stories::record` then carried it forward as precedent into
/// every prompt for the next ~8 pages -- teaching the model, at the top of its
/// context, that this is what a translation looks like here.
///
/// **Length is the wrong discriminator and was the first thing tried.**
/// Legitimate narration on the same volume reaches 724 characters -- an
/// encyclopedia-style caption -- so any cap that keeps those sits close enough
/// to a 2,131-character loop to be a coin toss. Repetition separates the two
/// cleanly where length does not.
///
/// Deliberately NOT gated on `misses.truncated`, though that flag is in scope at
/// the call site. Truncation is already handled: `prompt::translations`
/// seeds its result from the source, so a
/// segment the reply never reached comes back holding the Japanese and the
/// `source != translation` filter above drops it. And the relationship runs the
/// other way in any case -- that loop is what *caused* the truncation by
/// eating the token budget, so it is present on pages that never report one.
///
/// Case-folded because a mixed-case loop is the same defect; the allocation is
/// bounded by `max_tokens` and only reached by a pair long enough to judge.
fn is_degenerate_repetition(translation: &str) -> bool {
    let words = translation.split_whitespace().collect::<Vec<_>>();
    if words.len() < REPETITION_MIN_WORDS {
        return false;
    }
    let distinct = words
        .iter()
        .map(|word| word.to_lowercase())
        .collect::<std::collections::HashSet<_>>();
    // Multiplied out rather than divided, so an empty `words` could not divide
    // by zero even if the length guard above were ever relaxed.
    (distinct.len() as f32) < REPETITION_MIN_UNIQUE * words.len() as f32
}

/// Rejects an id that is empty, over-long, or not plain ASCII.
///
/// The id reaches a `HashMap` key and a response header, so it is bounded and
/// restricted at the edge rather than sanitised later. Hyphens are allowed
/// because the popup mints a UUID.
#[must_use]
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_ID_LEN
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(stories: &Stories, id: &str) -> Vec<(String, String)> {
        stories
            .context(id)
            .iter()
            .map(|pair| (pair.source.clone(), pair.translation.clone()))
            .collect()
    }

    #[test]
    fn an_unknown_story_carries_nothing_rather_than_failing() {
        let stories = Stories::default();
        assert!(stories.context("never-seen").is_empty());
        assert_eq!(stories.len("never-seen"), 0);
    }

    #[test]
    fn pairs_accumulate_in_reading_order() {
        let stories = Stories::default();
        stories.record("s", [("あ".to_owned(), "Ah".to_owned())]);
        stories.record("s", [("い".to_owned(), "Ee".to_owned())]);
        assert_eq!(
            pairs(&stories, "s"),
            [
                ("あ".to_owned(), "Ah".to_owned()),
                ("い".to_owned(), "Ee".to_owned())
            ]
        );
    }

    #[test]
    fn an_untranslated_segment_is_never_carried_forward() {
        /* A segment the model skipped comes back holding its source verbatim.
         * Teaching the next page that Japanese-in-Japanese-out is acceptable
         * would turn one miss into a habit. */
        let stories = Stories::default();
        stories.record(
            "s",
            [
                ("同じ".to_owned(), "同じ".to_owned()),
                ("  ".to_owned(), "blank source".to_owned()),
                ("ok".to_owned(), "".to_owned()),
                ("だめ".to_owned(), "No good".to_owned()),
            ],
        );
        assert_eq!(pairs(&stories, "s"), [("だめ".to_owned(), "No good".to_owned())]);
    }

    /// The runaway that cost a test volume its slowest page must not become
    /// precedent for the pages after it.
    #[test]
    fn a_degenerate_repetition_is_never_carried_forward() {
        let stories = Stories::default();
        // The measured shape of the runaway region, scaled down: one word,
        // repeated, long enough to be judged.
        let loop_text = "BEEP ".repeat(60).trim_end().to_owned();
        stories.record(
            "s",
            [
                ("ピピピ".to_owned(), loop_text),
                ("だめ".to_owned(), "No good".to_owned()),
            ],
        );
        assert_eq!(pairs(&stories, "s"), [("だめ".to_owned(), "No good".to_owned())]);
    }

    /// The two properties the threshold rests on, asserted rather than described.
    ///
    /// Short repetition is ordinary lettering and must survive; long narration is
    /// the population a naive length cap would have destroyed, and it must
    /// survive too. Between them they are why the rule is about repetition and
    /// not about size.
    #[test]
    fn repetition_is_judged_only_where_the_measurement_says_it_is_safe() {
        // Under the word floor: a laugh, a heartbeat, a running effect.
        for short in ["HA HA HA", "BEEP BEEP BEEP BEEP", "thump thump thump thump"] {
            assert!(
                !is_degenerate_repetition(short),
                "{short:?} is ordinary lettering"
            );
        }

        // Long and varied: the 724-character encyclopedia-style caption on the
        // test volume is the real shape here, and it scored 0.81 against a 0.20
        // threshold.
        let narration = "Many old customs survive here, such as the evening \
             bell and the lantern walk, and the town has long been known for the \
             gap between its quiet streets and the noise of its market days."
            .to_owned();
        assert!(!is_degenerate_repetition(&narration));

        // Case is not a hiding place.
        assert!(is_degenerate_repetition(&"Beep BEEP beep ".repeat(20)));

        /* The separation the threshold sits inside, so a future edit that moves
         * it has to move it past real data rather than past a hunch. Measured
         * over 952 pairs of >= 12 words drawn from six test runs: the four
         * degenerate ones score 0.002-0.005 and the next lowest legitimate pair
         * scores 0.714. */
        assert!(REPETITION_MIN_UNIQUE > 0.005 * 4.0, "too close to the loops");
        assert!(REPETITION_MIN_UNIQUE < 0.714 / 2.0, "too close to real prose");
    }

    #[test]
    fn the_window_keeps_the_newest_pairs() {
        let stories = Stories::default();
        for index in 0..DEFAULT_PAIRS + 5 {
            stories.record("s", [(format!("j{index}"), format!("e{index}"))]);
        }
        let kept = pairs(&stories, "s");
        assert_eq!(kept.len(), DEFAULT_PAIRS);
        // The oldest five are gone and the newest is still there.
        assert_eq!(kept[0].0, "j5");
        assert_eq!(kept[DEFAULT_PAIRS - 1].0, format!("j{}", DEFAULT_PAIRS + 4));
    }

    /// The window is the store's own field, not the compiled constant. As a
    /// `const`, every arm of a window-size comparison would need a rebuild --
    /// and a rebuild that is forgotten is how a stale release binary ignores a
    /// whole feature for an entire measurement run.
    #[test]
    fn the_configured_window_is_what_trims_rather_than_the_default() {
        let stories = Stories::with_window(3);
        for index in 0..10 {
            stories.record("s", [(format!("j{index}"), format!("e{index}"))]);
        }
        let kept = pairs(&stories, "s");
        assert_eq!(kept.len(), 3);
        assert_eq!(kept[0].0, "j7");
        assert_eq!(kept[2].0, "j9");
    }

    /// `--story-pairs 0` is the no-story arm, and it has to be reachable from the
    /// server rather than only by the client withholding an id: the extension
    /// always sends one, so this is the only way to run both arms against an
    /// unchanged browser.
    #[test]
    fn a_zero_window_carries_nothing_and_stores_nothing() {
        let stories = Stories::with_window(0);
        stories.record("s", [("あ".to_owned(), "Ah".to_owned())]);
        assert!(stories.context("s").is_empty());
        assert_eq!(stories.len("s"), 0);
        // Nothing was stored, so "New story" has nothing to forget either.
        assert!(!stories.clear("s"));
    }

    #[test]
    fn a_new_story_forgets_the_previous_one() {
        let stories = Stories::default();
        stories.record("old", [("あ".to_owned(), "Ah".to_owned())]);
        assert!(stories.clear("old"));
        assert!(stories.context("old").is_empty());
        // Clearing twice is not an error; the popup may retry.
        assert!(!stories.clear("old"));
    }

    #[test]
    fn stories_are_bounded_so_stale_ids_cannot_grow_without_limit() {
        let stories = Stories::default();
        for index in 0..MAX_STORIES + 4 {
            stories.record(&format!("s{index}"), [("あ".to_owned(), "Ah".to_owned())]);
        }
        let live = (0..MAX_STORIES + 4)
            .filter(|index| stories.len(&format!("s{index}")) > 0)
            .count();
        assert!(live <= MAX_STORIES, "{live} stories survived");
    }

    #[test]
    fn ids_are_bounded_and_ascii() {
        assert!(valid_id("01234567-89ab-7cde-8f01-23456789abcd"));
        assert!(valid_id("story_1"));
        assert!(!valid_id(""));
        assert!(!valid_id(&"a".repeat(MAX_ID_LEN + 1)));
        // Would otherwise reach a response header value.
        assert!(!valid_id("bad\r\nX-Injected: 1"));
        assert!(!valid_id("ストーリー"));
        assert!(!valid_id("../../etc"));
    }
}
