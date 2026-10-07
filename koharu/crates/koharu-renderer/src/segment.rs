//! Unicode line breaking with optional Chinese segmentation and hyphenation.

use std::{ops::Range, sync::LazyLock};

use hypher::{Lang, hyphenate_bounded};
use icu_properties::{
    CodePointMapData,
    props::{LineBreak, Script as IcuScript},
};
use icu_segmenter::{LineSegmenter, LineSegmenterBorrowed, options::LineBreakOptions};
use jieba_rs::Jieba;

static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);

/// Fewest characters a discretionary break may leave on either side.
///
/// See the rule that uses it in `LineBreaker::hyphenated_segments`.
const MIN_HYPHEN_FRAGMENT: usize = 3;

/// A line break candidate with its byte offset and whether it is mandatory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineBreakOpportunity {
    pub offset: usize,
    pub is_mandatory: bool,
}

/// Synthetic suffix to render only when a line actually breaks here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineBreakSuffix {
    Hyphen,
}

impl LineBreakSuffix {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hyphen => "-",
        }
    }
}

/// A trimmed line segment ready for shaping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineSegment {
    /// Range of visible text for this segment, excluding trailing mandatory break chars.
    pub range: Range<usize>,
    /// Byte offset where the next segment begins in the original string.
    pub next_offset: usize,
    /// Whether this segment ends with a mandatory break in the original text.
    pub is_mandatory: bool,
    /// Suffix to draw if this segment is the final segment on a wrapped line.
    pub break_suffix: Option<LineBreakSuffix>,
}

#[derive(Clone, Copy, Debug)]
struct HyphenationConfig {
    lang: Lang,
    min_word_len: usize,
}

/// Line breaker using ICU4X.
pub struct LineBreaker {
    segmenter: LineSegmenterBorrowed<'static>,
    hyphenation: Option<HyphenationConfig>,
    chinese_word_segmentation: bool,
}

fn trim_mandatory_break_suffix(text: &str, start: usize, end: usize) -> usize {
    let mut trimmed_end = end;
    while trimmed_end > start {
        let Some(ch) = text[..trimmed_end].chars().next_back() else {
            break;
        };
        if !matches!(ch, '\n' | '\r' | '\u{0085}' | '\u{2028}' | '\u{2029}') {
            break;
        }
        trimmed_end -= ch.len_utf8();
    }
    trimmed_end
}

impl LineBreaker {
    /// Creates a line breaker with ICU's default rules.
    #[must_use]
    pub fn new() -> Self {
        Self {
            segmenter: LineSegmenter::new_auto(LineBreakOptions::default()),
            hyphenation: None,
            chinese_word_segmentation: false,
        }
    }

    /// Keep Chinese Jieba words together when selecting discretionary line breaks.
    ///
    /// ICU still supplies the base Unicode line-break rules; this pass only
    /// removes non-mandatory break opportunities that fall inside likely
    /// Chinese word tokens.
    #[must_use]
    pub fn with_chinese_word_segmentation(mut self) -> Self {
        self.chinese_word_segmentation = true;
        self
    }

    /// Enable discretionary word hyphenation for long Latin words.
    ///
    /// `min_word_len` follows MangaTranslator's default threshold: short words
    /// keep ICU's normal break behavior, while long words gain extra break
    /// opportunities inside the word.
    #[must_use]
    pub fn with_hyphenation(mut self, lang: Lang, min_word_len: usize) -> Self {
        self.hyphenation = Some(HyphenationConfig { lang, min_word_len });
        self
    }

    /// Enable discretionary hyphenation from a BCP-47-ish language tag
    /// supported by `hypher`.
    #[must_use]
    pub fn with_hyphenation_tag(mut self, tag: &str, min_word_len: usize) -> Self {
        self.hyphenation =
            hyphenation_lang_from_tag(tag).map(|lang| HyphenationConfig { lang, min_word_len });
        self
    }

    /// Returns a vector of line break opportunities in the given text.
    pub fn line_break_opportunities(&self, text: &str) -> Vec<LineBreakOpportunity> {
        let opportunities = self
            .segmenter
            .segment_str(text)
            .map(|break_pos| LineBreakOpportunity {
                offset: break_pos,
                is_mandatory: text[..break_pos].chars().next_back().is_some_and(|c| {
                    matches!(
                        CodePointMapData::<LineBreak>::new().get(c),
                        LineBreak::MandatoryBreak
                            | LineBreak::CarriageReturn
                            | LineBreak::LineFeed
                            | LineBreak::NextLine
                    )
                }),
            })
            .collect();

        if self.chinese_word_segmentation && should_segment_as_chinese(text) {
            apply_chinese_word_segmentation(text, opportunities)
        } else {
            opportunities
        }
    }

    /// Returns shaped-text segments where mandatory break characters are excluded
    /// from the segment range but preserved in `next_offset`.
    pub fn line_segments(&self, text: &str) -> Vec<LineSegment> {
        self.line_break_opportunities(text)
            .windows(2)
            .flat_map(|window| {
                let start = window[0].offset;
                let end = window[1].offset;
                let is_mandatory = window[1].is_mandatory;
                let segment_end = if is_mandatory {
                    trim_mandatory_break_suffix(text, start, end)
                } else {
                    end
                };
                let segment = LineSegment {
                    range: start..segment_end,
                    next_offset: end,
                    is_mandatory,
                    break_suffix: None,
                };
                self.hyphenated_segments(text, segment)
            })
            .collect()
    }

    fn hyphenated_segments(&self, text: &str, segment: LineSegment) -> Vec<LineSegment> {
        let Some(config) = self.hyphenation else {
            return vec![segment];
        };
        if segment.is_mandatory || segment.range.is_empty() {
            return vec![segment];
        }

        let segment_text = &text[segment.range.clone()];
        let Some((core_start, core_end)) = hyphenatable_word_bounds(segment_text) else {
            return vec![segment];
        };

        let core = &segment_text[core_start..core_end];
        if core.chars().count() < config.min_word_len {
            return vec![segment];
        }
        /* Never break a proper noun. `hypher` is a Knuth-Liang automaton over
         * TeX's American-English patterns and lowercases its own buffer, so a
         * romanised Japanese name is hyphenated as though it were an English
         * word -- the dictionary's honest answer to a question it was never
         * asked. A reader reads that as a mistake, where
         * "EXPERI-ENCE" and "SEA-SONED" are merely ugly.
         *
         * Capitalisation is the only signal available at this layer -- there is
         * no part-of-speech tagging here and `Typography` carries nothing.
         *
         * A mid-sentence capital in translated dialogue is nearly always a
         * name, so there capitalisation is signal enough on its own.
         *
         * SENTENCE-INITIAL capitals are the hard case and the reason this guard
         * missed the surnames the blind panel found. A speech balloon is one or
         * two utterances, so its first word is nearly always sentence-initial,
         * and the manga convention of opening a line by naming the person
         * addressed puts a surname in exactly that slot: measured on the real
         * strings (names changed here), "I told Takahashi about it." was
         * protected while "Takahashi, what are you doing?", "Wait. Kitagawa
         * knows." and a line opening on a quotation mark were all shredded. In
         * this input the exemption is the common case, not the exception.
         *
         * Refusing every sentence-initial capital is NOT the fix, and the
         * measurement is unambiguous about it: over 20 pages that would also
         * refuse "Unforgivable!!", "Everyone...!" and "Anything~", and rendered
         * those balloons at 16px, 12px and 20px where a break had let them fill
         * a burst balloon at 28px, 22px and 29px. Small text floating in a
         * shout balloon is a worse fault than the one being fixed.
         *
         * So sentence-initially the guard asks the narrower question it always
         * meant to ask -- is this a *Japanese* word wearing English
         * hyphenation patterns -- and `is_romanised_japanese` answers it. On
         * the measured pages that separates the two classes exactly: all three
         * name instances scan, and all twelve ordinary words do not.
         *
         * Note this reads the ORIGINAL text, not the uppercased one. Text that
         * is entirely capitals carries no signal at all, which is one more
         * reason `--uppercase-dialogue` stays off. */
        if starts_uppercase(core)
            && (!is_sentence_initial(text, segment.range.start + core_start)
                || is_romanised_japanese(core))
        {
            return vec![segment];
        }
        /* Never put a second break inside a word that already carries a hyphen.
         * ICU offers a line break after the literal hyphen, so "self-conscious"
         * reaches this function as the two segments "self-" and "conscious",
         * and the second is long enough to be hyphenated on its own --
         * "self-con-scious", two hyphens in one word. Looking at the whole
         * whitespace-delimited word is what makes the rule statable: neither
         * half can see the other. */
        if word_is_already_hyphenated(text, segment.range.start + core_start, core.len()) {
            return vec![segment];
        }

        /* No fragment shorter than three characters either side of a break.
         * `hypher`'s English bounds are (2, 3), so it will break after two
         * letters -- "Fu-jinami", "Ex-perience", "un-derstand". A two-character
         * fragment is the most conspicuous fault the blind panel reported, and
         * it costs almost nothing to forbid: the word breaks one syllable later
         * instead ("Expe-rience"), or not at all. `max` rather than an
         * assignment, so a language whose own bounds are wider keeps them. */
        let (left_min, right_min) = config.lang.bounds();
        let syllables: Vec<&str> = hyphenate_bounded(
            core,
            config.lang,
            left_min.max(MIN_HYPHEN_FRAGMENT),
            right_min.max(MIN_HYPHEN_FRAGMENT),
        )
        .collect();
        if syllables.len() <= 1 {
            return vec![segment];
        }

        let mut result = Vec::with_capacity(syllables.len());
        let mut word_offset = 0usize;
        for (idx, syllable) in syllables.iter().enumerate() {
            let is_last = idx + 1 == syllables.len();
            let start = if idx == 0 {
                segment.range.start
            } else {
                segment.range.start + core_start + word_offset
            };
            word_offset += syllable.len();
            let end = if is_last {
                segment.range.end
            } else {
                segment.range.start + core_start + word_offset
            };
            result.push(LineSegment {
                range: start..end,
                next_offset: if is_last { segment.next_offset } else { end },
                is_mandatory: false,
                break_suffix: (!is_last).then_some(LineBreakSuffix::Hyphen),
            });
        }

        result
    }
}

impl Default for LineBreaker {
    fn default() -> Self {
        Self::new()
    }
}

pub fn hyphenation_lang_from_tag(value: &str) -> Option<Lang> {
    let lower = value.trim().to_ascii_lowercase();
    let primary = lower
        .split(['-', '_'])
        .next()
        .filter(|part| part.len() == 2)?;
    Lang::from_iso(primary.as_bytes().try_into().ok()?)
}

fn apply_chinese_word_segmentation(
    text: &str,
    opportunities: Vec<LineBreakOpportunity>,
) -> Vec<LineBreakOpportunity> {
    let protected_ranges = chinese_word_ranges(text);
    if protected_ranges.is_empty() {
        return opportunities;
    }

    opportunities
        .into_iter()
        .filter(|opportunity| {
            opportunity.is_mandatory
                || !protected_ranges
                    .iter()
                    .any(|range| opportunity.offset > range.start && opportunity.offset < range.end)
        })
        .collect()
}

fn chinese_word_ranges(text: &str) -> Vec<Range<usize>> {
    let mut offset = 0usize;
    JIEBA
        .cut(text, true)
        .into_iter()
        .filter_map(|token| {
            let word = token.word;
            let start = offset;
            let end = start + word.len();
            offset = end;

            is_chinese_word_token(word).then_some(start..end)
        })
        .collect()
}

fn should_segment_as_chinese(text: &str) -> bool {
    let script_map = CodePointMapData::<IcuScript>::new();
    let mut has_chinese = false;

    for ch in text.chars() {
        match script_map.get(ch) {
            IcuScript::Han | IcuScript::Bopomofo => has_chinese = true,
            IcuScript::Hiragana | IcuScript::Katakana => return false,
            _ => {}
        }
    }

    has_chinese
}

fn is_chinese_word_token(word: &str) -> bool {
    let script_map = CodePointMapData::<IcuScript>::new();
    let mut has_chinese = false;
    let mut char_count = 0usize;

    for ch in word.chars() {
        char_count += 1;
        match script_map.get(ch) {
            IcuScript::Han | IcuScript::Bopomofo => has_chinese = true,
            IcuScript::Hiragana | IcuScript::Katakana | IcuScript::Hangul => return false,
            _ => {}
        }
    }

    has_chinese && char_count > 1
}

/// Whether a word's first letter is a capital.
fn starts_uppercase(core: &str) -> bool {
    core.chars().next().is_some_and(char::is_uppercase)
}

/// Whether the word beginning at `start` opens a sentence.
///
/// Looks backwards past whitespace and opening punctuation for a terminator.
/// Nothing before it -- the beginning of the text -- counts as sentence-initial
/// too.
fn is_sentence_initial(text: &str, start: usize) -> bool {
    const OPENERS: [char; 6] = ['"', '\'', '(', '[', '\u{201c}', '\u{2018}'];
    const TERMINATORS: [char; 6] = ['.', '!', '?', '\u{2026}', ':', ';'];

    let mut seen_gap = false;
    for character in text[..start].chars().rev() {
        if character.is_whitespace() || OPENERS.contains(&character) {
            seen_gap = true;
            continue;
        }
        // A terminator immediately abutting the word is an abbreviation dot, not
        // a sentence end -- so the gap is required.
        return seen_gap && TERMINATORS.contains(&character);
    }
    true
}

/// Consonant clusters Hepburn romanisation can put in front of a vowel.
///
/// The empty string is in the set because a mora may be a bare vowel. What is
/// *absent* is doing most of the work: no `l`, `q`, `v`, `x`, no bare `c`, and
/// no cluster outside this list, which is what an English word reliably
/// contains and a romanised Japanese one cannot.
const HEPBURN_ONSETS: [&str; 33] = [
    "", "k", "g", "s", "z", "t", "d", "n", "h", "b", "p", "m", "y", "r", "w", "f", "j", "sh", "ch",
    "ts", "ky", "gy", "ny", "hy", "by", "py", "my", "ry", "sy", "ty", "dy", "zy", "jy",
];

/// Whether a word scans as romanised Japanese.
///
/// `hypher` is a Knuth-Liang automaton over TeX's American-English patterns and
/// it lowercases its own buffer, so it will hyphenate anything spelled in Latin
/// letters, a romanised Japanese name included -- its honest answer to a
/// question it was never asked. This is that question: does the word decompose into Hepburn morae,
/// each an allowed onset plus a vowel, with only a syllabic `n` or a geminate
/// consonant outside that shape?
///
/// It is a *phonotactic* test, not a dictionary, so English words that happen
/// to fit the shape -- "Separate", "Moderate", "Panorama" -- are refused a break
/// when they open a sentence. That is the same bounded cost the capitalisation
/// guard already accepts mid-sentence, and it costs at most one size step.
fn is_romanised_japanese(word: &str) -> bool {
    const VOWELS: [char; 5] = ['a', 'i', 'u', 'e', 'o'];
    /// Below this a word is too short to be evidence of anything: "Also" and
    /// "Same" scan, and `min_word_len` means nothing this short reaches here
    /// anyway.
    const MINIMUM: usize = 4;

    let lowered = word.to_lowercase();
    let letters: Vec<char> = lowered.chars().collect();
    if letters.len() < MINIMUM {
        return false;
    }
    let mut index = 0;
    let mut morae = 0;
    while index < letters.len() {
        let letter = letters[index];
        // A geminate belongs to the mora after it -- "Sakki", "Itchi".
        if !VOWELS.contains(&letter)
            && letter != 'n'
            && letters.get(index + 1) == Some(&letter)
        {
            if !HEPBURN_ONSETS.contains(&letter.to_string().as_str()) {
                return false;
            }
            index += 1;
            continue;
        }
        // Syllabic n: an `n` with no vowel (or palatal glide) after it.
        if letter == 'n'
            && letters
                .get(index + 1)
                .is_none_or(|next| !VOWELS.contains(next) && *next != 'y')
        {
            index += 1;
            morae += 1;
            continue;
        }
        let onset_start = index;
        while index < letters.len() && !VOWELS.contains(&letters[index]) {
            index += 1;
            if index - onset_start > 2 {
                return false;
            }
        }
        if index == letters.len() {
            // A trailing consonant that is not a syllabic n. English, not
            // Japanese.
            return false;
        }
        let onset: String = letters[onset_start..index].iter().collect();
        if !HEPBURN_ONSETS.contains(&onset.as_str()) {
            return false;
        }
        index += 1;
        morae += 1;
    }
    morae >= 2
}

/// Whether `character` is a hyphen that is part of a word rather than a dash
/// between them.
///
/// An en or em dash is punctuation separating words and ICU already breaks
/// around it, so including either here would refuse hyphenation for both
/// neighbours of every dash.
const fn is_word_hyphen(character: char) -> bool {
    matches!(character, '-' | '\u{2010}' | '\u{2011}')
}

/// Whether the whitespace-delimited word covering `start..start + len` already
/// contains a literal hyphen.
///
/// The core alone cannot answer this: ICU hands out "self-" and "conscious" as
/// two segments, so the half being considered for a discretionary break never
/// contains the hyphen that disqualifies it. Widening to the surrounding
/// non-whitespace run is what lets each half see the other.
fn word_is_already_hyphenated(text: &str, start: usize, len: usize) -> bool {
    let before = text[..start]
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or_default();
    let after = text[start + len..]
        .split(char::is_whitespace)
        .next()
        .unwrap_or_default();
    before.chars().chain(after.chars()).any(is_word_hyphen)
}

fn hyphenatable_word_bounds(text: &str) -> Option<(usize, usize)> {
    let start = text.find(|ch: char| ch.is_alphabetic())?;
    let end = text
        .char_indices()
        .rev()
        .find(|&(_, ch)| ch.is_alphabetic())
        .map(|(idx, ch)| idx + ch.len_utf8())?;
    if start >= end {
        return None;
    }

    let core = &text[start..end];
    core.chars()
        .all(|ch| ch.is_alphabetic())
        .then_some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn break_on_whitespace() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let linebreaker = LineBreaker::new();
        let breaks = linebreaker.line_break_opportunities(text);
        let segments: Vec<&str> = breaks
            .windows(2)
            .map(|w| &text[w[0].offset..w[1].offset])
            .collect();
        let expected = vec![
            "The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog.",
        ];
        assert_eq!(segments, expected);
    }

    #[test]
    fn break_on_newline() {
        let text = "Hello, \nWorld!";
        let linebreaker = LineBreaker::new();
        let breaks = linebreaker.line_break_opportunities(text);
        let expected = vec![
            LineBreakOpportunity {
                offset: 0,
                is_mandatory: false,
            },
            LineBreakOpportunity {
                offset: 8,
                is_mandatory: true,
            },
            LineBreakOpportunity {
                offset: 14,
                is_mandatory: false,
            },
        ];
        assert_eq!(breaks, expected);
    }

    #[test]
    fn line_segments_trim_newline_suffixes() {
        let text = "Hello, \nWorld!";
        let linebreaker = LineBreaker::new();
        let segments = linebreaker.line_segments(text);

        assert_eq!(segments.len(), 2);
        assert_eq!(&text[segments[0].range.clone()], "Hello, ");
        assert_eq!(segments[0].next_offset, 8);
        assert!(segments[0].is_mandatory);
        assert_eq!(segments[0].break_suffix, None);
        assert_eq!(&text[segments[1].range.clone()], "World!");
        assert_eq!(segments[1].next_offset, text.len());
        assert!(!segments[1].is_mandatory);
        assert_eq!(segments[1].break_suffix, None);
    }

    #[test]
    fn chinese_word_segmentation_keeps_jieba_words_together() {
        let text = "\u{5357}\u{4eac}\u{5e02}\u{957f}\u{6c5f}\u{5927}\u{6865}";
        let linebreaker = LineBreaker::new().with_chinese_word_segmentation();
        let segments: Vec<&str> = linebreaker
            .line_segments(text)
            .iter()
            .map(|segment| &text[segment.range.clone()])
            .collect();

        assert_eq!(
            segments,
            vec![
                "\u{5357}\u{4eac}\u{5e02}",
                "\u{957f}\u{6c5f}\u{5927}\u{6865}",
            ]
        );
    }

    #[test]
    fn chinese_word_segmentation_preserves_icu_punctuation_rules() {
        let text = "\u{5c0f}\u{8bf4}\u{ff0c}\u{4f60}\u{597d}";
        let linebreaker = LineBreaker::new().with_chinese_word_segmentation();
        let segments: Vec<&str> = linebreaker
            .line_segments(text)
            .iter()
            .map(|segment| &text[segment.range.clone()])
            .collect();

        assert_eq!(
            segments,
            vec!["\u{5c0f}\u{8bf4}\u{ff0c}", "\u{4f60}\u{597d}",]
        );
    }

    #[test]
    fn chinese_word_segmentation_does_not_resegment_kana_text() {
        let text = "\u{543e}\u{8f29}\u{306f}\u{732b}";
        let linebreaker = LineBreaker::new().with_chinese_word_segmentation();
        let segments: Vec<&str> = linebreaker
            .line_segments(text)
            .iter()
            .map(|segment| &text[segment.range.clone()])
            .collect();

        assert_eq!(
            segments,
            vec!["\u{543e}", "\u{8f29}", "\u{306f}", "\u{732b}",]
        );
    }

    #[test]
    fn hyphenation_adds_discretionary_segments_to_long_latin_words() {
        let text = "antidisestablishmentarianism";
        let linebreaker = LineBreaker::new().with_hyphenation(Lang::English, 8);
        let segments = linebreaker.line_segments(text);

        assert!(
            segments.len() > 1,
            "expected long word to be split into hyphenation segments, got {segments:?}"
        );
        for segment in segments.iter().take(segments.len() - 1) {
            assert_eq!(segment.break_suffix, Some(LineBreakSuffix::Hyphen));
            assert!(!segment.is_mandatory);
        }
        assert_eq!(segments.last().unwrap().break_suffix, None);

        let rebuilt = segments
            .iter()
            .map(|segment| &text[segment.range.clone()])
            .collect::<String>();
        assert_eq!(rebuilt, text);
    }

    /// The visible pieces of `text`, paired with whether a break there draws a
    /// hyphen.
    fn hyphenated_pieces<'a>(
        breaker: &LineBreaker,
        text: &'a str,
    ) -> Vec<(&'a str, bool)> {
        breaker
            .line_segments(text)
            .into_iter()
            .map(|segment| {
                (
                    &text[segment.range.clone()],
                    segment.break_suffix.is_some(),
                )
            })
            .collect()
    }

    #[test]
    fn hyphenation_never_breaks_a_name_wherever_it_sits() {
        /* The failure the blind panel reported, in the four positions a name
         * actually occupies. Only the first was protected before: the guard
         * exempted sentence-initial words, and a speech balloon opening on the
         * name of the person being addressed is sentence-initial by
         * construction, so the exemption covered exactly the manga case. */
        let breaker = LineBreaker::new().with_hyphenation(Lang::English, 8);
        for text in [
            "I told Takahashi about it.",
            "Takahashi, what are you doing?",
            "Wait. Kitagawa knows.",
            "\u{201c}Takahashi is here.\u{201d}",
            "TAKAHASHI",
            "Hoshizora",
        ] {
            let pieces = hyphenated_pieces(&breaker, text);
            assert!(
                pieces.iter().all(|&(_, hyphenated)| !hyphenated),
                "a name was broken in {text:?}: {pieces:?}"
            );
        }

        // The guard must still be about capitals, not about hyphenation as
        // such: the same words in lower case break exactly as before.
        let pieces = hyphenated_pieces(&breaker, "i told takahashi about it.");
        assert!(
            pieces.iter().any(|&(_, hyphenated)| hyphenated),
            "expected the lower-case control to still hyphenate: {pieces:?}"
        );
    }

    #[test]
    fn an_ordinary_word_opening_a_sentence_still_breaks() {
        /* Measured, and the reason the guard is not simply "never break a
         * capital": rendered on real pages, refusing these three put 16px,
         * 12px and 20px of text in burst balloons that a break had filled at
         * 28px, 22px and 29px. Every word here is one the blind panel would
         * have been handed if the guard were widened without this test. */
        let breaker = LineBreaker::new().with_hyphenation(Lang::English, 8);
        for text in [
            "Unforgivable!!",
            "Everyone...!",
            "Anything~",
            "Automobiles",
            "Chattering",
            "Originally founded as an institution",
            "Honestly",
            "Afternoon",
        ] {
            let pieces = hyphenated_pieces(&breaker, text);
            assert!(
                pieces.iter().any(|&(_, hyphenated)| hyphenated),
                "an ordinary word lost its break in {text:?}: {pieces:?}"
            );
        }
    }

    #[test]
    fn a_romanised_name_is_told_from_an_english_word() {
        /* The whole of rule 1's discrimination, over exactly the words the
         * 20-page measurement produced: three name instances on one side, the
         * twelve sentence-initial ordinary words on the other. The names are
         * stand-ins of the same lengths for the measured ones. */
        for name in [
            "Takahashi", "Hoshizora", "Kitagawa", "Yamamoto", "Morimoto", "Minori", "Tomoya",
        ] {
            assert!(is_romanised_japanese(name), "{name} should scan");
        }
        for english in [
            "Originally",
            "Everyone",
            "Railroads",
            "Automobiles",
            "Chattering",
            "Honestly",
            "Cheapskate",
            "Anything",
            "Unforgivable",
            "Everything",
            "Targeting",
            "Afternoon",
        ] {
            assert!(!is_romanised_japanese(english), "{english} should not scan");
        }
        // The shapes the scanner exists to reject, one each: a consonant
        // Hepburn has no letter for, a cluster it cannot form, and a word
        // ending on a consonant that is not a syllabic n.
        assert!(!is_romanised_japanese("velocity"));
        assert!(!is_romanised_japanese("nostalgia"));
        assert!(!is_romanised_japanese("sakurab"));
        // ...against the shapes it must accept: a syllabic n, a geminate, and a
        // long vowel.
        assert!(is_romanised_japanese("kanpai"));
        assert!(is_romanised_japanese("gakkou"));
        assert!(is_romanised_japanese("ojousama"));
    }

    #[test]
    fn hyphenation_leaves_three_characters_on_both_sides_of_a_break() {
        /* A break is taken at one syllable boundary, so the fragments a reader
         * sees are the whole prefix and the whole suffix -- which makes the
         * shortest possible fragment the first syllable and the last. */
        let breaker = LineBreaker::new().with_hyphenation(Lang::English, 8);
        for word in [
            // hypher's raw (2, 3) bounds give "fu-ji-nami"; the guard gives "fuji-nami".
            "fujinami",
            "experience",
            "understand",
            "president",
            "everything",
            "antidisestablishmentarianism",
        ] {
            let pieces = hyphenated_pieces(&breaker, word);
            if pieces.len() < 2 {
                continue;
            }
            assert!(
                pieces[0].0.chars().count() >= MIN_HYPHEN_FRAGMENT,
                "leading fragment of {word:?} is shorter than {MIN_HYPHEN_FRAGMENT}: {pieces:?}"
            );
            assert!(
                pieces[pieces.len() - 1].0.chars().count() >= MIN_HYPHEN_FRAGMENT,
                "trailing fragment of {word:?} is shorter than {MIN_HYPHEN_FRAGMENT}: {pieces:?}"
            );
        }
    }

    #[test]
    fn hyphenation_does_not_add_a_second_hyphen_to_a_hyphenated_word() {
        let breaker = LineBreaker::new().with_hyphenation(Lang::English, 8);
        for text in [
            "self-conscious",
            "well, self-conscious people do that.",
            "conscious-self",
        ] {
            let pieces = hyphenated_pieces(&breaker, text);
            assert!(
                pieces.iter().all(|&(_, hyphenated)| !hyphenated),
                "added a second hyphen in {text:?}: {pieces:?}"
            );
        }

        /* The control that makes the rule mean something: the same word breaks
         * when it is not half of a hyphenated compound, so the refusal above is
         * caused by the hyphen and not by the word being unbreakable. */
        let pieces = hyphenated_pieces(&breaker, "conscious");
        assert!(
            pieces.iter().any(|&(_, hyphenated)| hyphenated),
            "expected the unhyphenated control to break: {pieces:?}"
        );
    }

    #[test]
    fn hyphenation_language_tags_cover_hypher_languages() {
        let cases = [
            ("af", Lang::Afrikaans),
            ("sq", Lang::Albanian),
            ("as", Lang::Assamese),
            ("be", Lang::Belarusian),
            ("bn", Lang::Bengali),
            ("bg", Lang::Bulgarian),
            ("ca", Lang::Catalan),
            ("hr", Lang::Croatian),
            ("cs", Lang::Czech),
            ("da", Lang::Danish),
            ("nl", Lang::Dutch),
            ("en-US", Lang::English),
            ("et", Lang::Estonian),
            ("fi", Lang::Finnish),
            ("fr-FR", Lang::French),
            ("gl", Lang::Galician),
            ("ka", Lang::Georgian),
            ("de-DE", Lang::German),
            ("el", Lang::Greek),
            ("gu", Lang::Gujarati),
            ("hi", Lang::Hindi),
            ("hu", Lang::Hungarian),
            ("is", Lang::Icelandic),
            ("it-IT", Lang::Italian),
            ("kn", Lang::Kannada),
            ("ku", Lang::Kurmanji),
            ("la", Lang::Latin),
            ("lt", Lang::Lithuanian),
            ("ml", Lang::Malayalam),
            ("mr", Lang::Marathi),
            ("mn", Lang::Mongolian),
            ("no", Lang::Norwegian),
            ("nb", Lang::Norwegian),
            ("nn", Lang::Norwegian),
            ("or", Lang::Oriya),
            ("pa", Lang::Panjabi),
            ("pl", Lang::Polish),
            ("pt-BR", Lang::Portuguese),
            ("ru", Lang::Russian),
            ("sa", Lang::Sanskrit),
            ("sr", Lang::Serbian),
            ("sk", Lang::Slovak),
            ("sl", Lang::Slovenian),
            ("es-ES", Lang::Spanish),
            ("sv", Lang::Swedish),
            ("ta", Lang::Tamil),
            ("te", Lang::Telugu),
            ("tr", Lang::Turkish),
            ("tk", Lang::Turkmen),
            ("uk", Lang::Ukrainian),
        ];

        for (tag, lang) in cases {
            assert_eq!(hyphenation_lang_from_tag(tag), Some(lang), "tag={tag}");
        }

        assert_eq!(hyphenation_lang_from_tag("German"), None);
        assert_eq!(hyphenation_lang_from_tag("ja-JP"), None);
    }

    #[test]
    fn hyphenation_supports_unicode_words() {
        let text = "электрификация";
        let linebreaker = LineBreaker::new().with_hyphenation(Lang::Russian, 8);
        let segments = linebreaker.line_segments(text);

        assert!(
            segments.len() > 1,
            "expected unicode word to be split into hyphenation segments, got {segments:?}"
        );
        let rebuilt = segments
            .iter()
            .map(|segment| &text[segment.range.clone()])
            .collect::<String>();
        assert_eq!(rebuilt, text);
    }

    #[test]
    fn japanese_break_on_characters() {
        let text = "吾輩は猫である。名前はまだない。";
        let linebreaker = LineBreaker::new();
        let breaks = linebreaker.line_break_opportunities(text);
        let segments: Vec<&str> = breaks
            .windows(2)
            .map(|w| &text[w[0].offset..w[1].offset])
            .collect();
        let expected = vec![
            "吾", "輩", "は", "猫", "で", "あ", "る。", "名", "前", "は", "ま", "だ", "な", "い。",
        ];
        assert_eq!(segments, expected);
    }

    #[test]
    fn mixed_language_breaks_01() {
        let text = "『シャイニング』（The Shining）は、スタンリー・キューブリックが製作・監督し、小説家のダイアン・ジョンソンと共同脚本を務めた、1980年公開のサイコロジカルホラー映画。";
        let linebreaker = LineBreaker::new();
        let breaks = linebreaker.line_break_opportunities(text);
        let segments: Vec<&str> = breaks
            .windows(2)
            .map(|w| &text[w[0].offset..w[1].offset])
            .collect();
        #[rustfmt::skip]
        let expected = vec![
            "『シャ", "イ", "ニ", "ン", "グ』", "（The ", "Shining）", "は、", "ス", "タ", "ン", "リー・", "キュー", "ブ", "リッ", "ク", "が", "製", "作・", "監", "督", "し、", "小", "説", "家", "の", "ダ", "イ", "ア", "ン・", "ジョ", "ン", "ソ", "ン", "と", "共", "同", "脚", "本", "を", "務", "め", "た、", "1980", "年", "公", "開", "の", "サ", "イ", "コ", "ロ", "ジ", "カ", "ル", "ホ", "ラー", "映", "画。"
        ];
        assert_eq!(segments, expected);
    }

    #[test]
    fn mixed_language_breaks_02() {
        let text = "《我是猫》是日本作家夏目漱石创作的长篇小说，也是其代表作，它确立了夏目漱石在文学史上的地位。作品淋漓尽致地反映了二十世纪初，日本中小资产阶级的思想和生活，尖锐地揭露和批判了明治“文明开化”的资本主义社会。小说采用幽默、讽刺、滑稽的手法，借助一只猫的视觉、听觉、感觉，嘲笑了明治时代知识分子空虚的精神生活，小说构思奇巧，描写夸张，结构灵活，具有鲜明的艺术特色。";
        let linebreaker = LineBreaker::new();
        let breaks = linebreaker.line_break_opportunities(text);
        let segments: Vec<&str> = breaks
            .windows(2)
            .map(|w| &text[w[0].offset..w[1].offset])
            .collect();
        #[rustfmt::skip]
        let expected = vec![
            "《我", "是", "猫》", "是", "日", "本", "作", "家", "夏", "目", "漱", "石", "创", "作", "的", "长", "篇", "小", "说，", "也", "是", "其", "代", "表", "作，", "它", "确", "立", "了", "夏", "目", "漱", "石", "在", "文", "学", "史", "上", "的", "地", "位。", "作", "品", "淋", "漓", "尽", "致", "地", "反", "映", "了", "二", "十", "世", "纪", "初，", "日", "本", "中", "小", "资", "产", "阶", "级", "的", "思", "想", "和", "生", "活，", "尖", "锐", "地", "揭", "露", "和", "批", "判", "了", "明", "治“文", "明", "开", "化”的", "资", "本", "主", "义", "社", "会。", "小", "说", "采", "用", "幽", "默、", "讽", "刺、", "滑", "稽", "的", "手", "法，", "借", "助", "一", "只", "猫", "的", "视", "觉、", "听", "觉、", "感", "觉，", "嘲", "笑", "了", "明", "治", "时", "代", "知", "识", "分", "子", "空", "虚", "的", "精", "神", "生", "活，", "小", "说", "构", "思", "奇", "巧，", "描", "写", "夸", "张，", "结", "构", "灵", "活，", "具", "有", "鲜", "明", "的", "艺", "术", "特", "色。"
        ];
        assert_eq!(segments, expected);
    }
}
