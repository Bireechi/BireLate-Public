//! Refuse to letter a region whose OCR text is not text.
//!
//! **The defect, verified in pixels.** A Chinese test page of falling petals
//! carries *no text on it at all*. Detection found 23 regions in the petal
//! shapes, PaddleOCR-VL returned one-to-four character strings for them --
//! including hallucinated Japanese kana on a Chinese page -- the translator
//! faithfully turned those into English, and the renderer lettered `PINK PIKA`,
//! `HI`, `HMPH`, `HITO` and `20000070` across clean artwork. `untranslated` was
//! empty, `truncated` was false, and every number a translator sweep produced
//! said the page was fine. It ran `format=json` and scored segments; nothing in
//! it ever looked at a rendered page.
//!
//! **The rules are not new.** An offline segment labeller classified exactly
//! this for the translation benchmark -- it is how the usable share of each test
//! source's segments is known. It ran over saved responses and could not affect
//! what a reader sees. This is that classifier, moved to where it can refuse.
//!
//! **It refuses free-standing text only, and that is measured rather than
//! cautious.** See `hide_implausible`: an erased balloon with nothing put back
//! is a worse defect than the beat it withholds, and 147 of the 161 phantom
//! segments are on artwork anyway.
//!
//! **Where it runs, and what that costs.** A scene patch between
//! `Pipeline::execute` and the render, exactly like `sfx.rs` and `lettering.rs`,
//! so it needs no koharu change and no pipeline reload. The price of that seam is
//! stated rather than hidden: by the time it runs, the LLM has already
//! translated the junk, **and the artwork underneath is already erased**. The
//! erase mask is built in the *detection* stage (`detection.rs:1884` writes the
//! `text-mask` asset) and inpainting only reads it, so no gate keyed on OCR text
//! can save those pixels -- inpainting and OCR are siblings under detection
//! (`scheduler.rs:145-151`) and run concurrently. Saving the artwork needs a
//! geometry-only rule at `mask_includes`, which is where the pipeline's own
//! mask filter already sits. This gate stops the nonsense being *drawn*; it does
//! not un-erase.
//!
//! **Hiding, not deleting, and that is load-bearing.** Clearing `Translation`
//! alone renders the raw OCR string instead: `RenderRequest` defaults
//! `fallback_to_source_text: true` (`koharu-renderer/src/request.rs:115`) and the
//! server never overrides it, so `resolve_text` falls back to `SourceText` and
//! the reader gets the hallucinated kana instead of the hallucinated English --
//! strictly worse. Setting `Visibility { visible: false }` on the layer drops it
//! in the compositor's traversal (`compositor.rs:342-361`) before any glyph is
//! shaped, and leaves the region in the scene, so `format=json` can still report
//! what was refused and why. A refusal that erased its own evidence would be
//! untestable from outside.
//!
//! **A refusal has TWO consequences, and for the drawn-connective rule both are
//! wanted.** `routes.rs:875-885` stamps the reason onto the region, which hides
//! the layer from the render; `routes.rs:899-905` then filters every refused
//! region out of the story window. For [`DRAWN_CONNECTIVES`] the first is the
//! whole point -- a full-panel "AND SO," across artwork. The second matters
//! just as much and is easy to miss: the story window is fed verbatim into the
//! next page's prompt for up to `--story-pairs` pages, so a `そして、 -> And so,`
//! pair harvested off artwork would be carried forward as *established
//! terminology* for the rest of the story. That is the same mechanism the
//! `ピンク -> PINK` note below describes, and it is worse here rather than
//! better, because a connective looks like a legitimate pair and so survives
//! every plausibility check a reader of the prompt might apply.
//!
//! **Validated against a frozen test corpus before it was written**: all 511
//! labelled segments, four sources, three languages. The rules
//! suppress **161 of 161** segments the labeller calls non-text and **1** of the
//! 350 it calls usable. Without a declared source language the script rule is
//! off and the figure is 122; the one suppression below is language-independent
//! and stands in both arms.
//!
//! **That one is [`DRAWN_CONNECTIVES`], and the labeller is what is wrong there
//! -- checked in pixels.** It is a region read `それは、`, `role: free-text`, box
//! (178.9, 742.6, 239.7, 230.4) on an 844x1200 page, and the labeller calls it
//! `label: "ok"`. The frozen corpus overwrites the detector's `label` with the
//! labeller's verdict, so the class has to be read off the same box in another
//! run, which has it byte-identical with `label: "onomatopoeia"`. Cropping those
//! 240x230 pixels out of the page settles it -- they are a large rough
//! brush-drawn effect, carrying no such text and no text at all. So this is a
//! false positive of the *labeller*, not of the rule, and the count is quoted as
//! 1 rather than 0 because a number that hides a disagreement is worth less than
//! one that names it. **`DrawnConnective` is the only rule here that can
//! suppress a segment the labeller calls usable**; the other five still stand at
//! zero, which is what makes them comparable to the labeller and this one not.
//!
//! **Writing this found a bug in the labeller it was ported from**, and the
//! numbers above are the corrected ones. `・` (U+30FB) sits inside the kana
//! block, so a range test that reaches the block first counts a row of
//! interpuncts as kana -- making it "100% foreign script" on a Korean page, the
//! right refusal for entirely the wrong reason, and no refusal at all on a
//! Japanese page where no script rule fires. Halfwidth katakana had the mirror
//! fault: `is_alphabetic` is true for them, so they counted as *Latin* and
//! escaped the one rule they should trip. Both are fixed on both sides, and
//! the labeller's own usable rates moved with it: 369 -> 350 segments,
//! ja 93% -> 91%, manhua 47% -> 40%, webtoon 70% -> 67%, manhwa 58% -> 54%.

use koharu_scene::{EntityId, Origin, Region, Session, Snapshot, TextLayout, Visibility};
use koharu_translator::Language;

/// The detector's class for a drawn sound effect. Matches `sfx.rs` and
/// `regions.rs` -- once effects are promoted to `TextRegion` so OCR will read
/// them, this label is the ONLY thing that still separates an effect from a line
/// of dialogue (see `RegionOut::label`).
const ONOMATOPOEIA: &str = "onomatopoeia";

/// Sentence-opening connectives that came back as the **entire** read of a drawn
/// sound effect. A CLOSED, ENUMERATED set, and every member was read off stored
/// runs rather than invented.
///
/// **The defect.** 9 of the 2,441 regions on a Japanese test volume -- 0.4%
/// overall but **1.4% of the 636 onomatopoeia regions** -- come back as a bare
/// connective where the page carries drawn artwork and no such text. All 9 were
/// checked against the printed page. The worst is a 1210x1247 `onomatopoeia` box
/// on an 844x1200 panel, read `そして、` and lettered as a full-panel **"AND
/// SO,"** across artwork carrying no text whatsoever.
///
/// The whole set, with row counts over every stored run and the unique regions
/// behind them. `text` is the count on regions the detector labelled `text`
/// rather than `onomatopoeia` -- the population this rule must not touch:
///
/// | source | ono rows | unique regions | text rows | rendered as |
/// |---|---|---|---|---|
/// | `それは、` | 21 | 2 | **0** | "That's...", "That is," |
/// | `そして、` | 13 | 3 | **0** | "And so,", "And then," |
/// | `しかし、` | 12 | 1 | **0** | "However," |
/// | `そういえば、` | 11 | 2 | **0** | "Come to think of it," |
/// | `それでも、` | 10 | 1 | **0** | "Even so," |
///
/// **Nine unique regions.** Region indices are **per run** -- two detections of
/// one page found 20 and 25 regions on it -- so only the geometry identifies a
/// region, and counting by index across runs once made the same region look like
/// two. The column sums 2+3+1+2+1 = 9, which is the nine that were hand-checked.
/// (Row counts drift upward as runs accumulate; the unique-region identity is
/// what the rule rests on and does not.)
///
/// **Three of the nine are already adjudicated by a second engine**: where one
/// engine read `そういえば、` once and `そして、` twice, the other read empty,
/// empty, and a `ピー` loop. In none does the second engine confirm the
/// connective.
///
/// **THE EMPTY BAND.** Across every stored run -- 24,281 region rows, 17,674 of
/// them labelled `text` -- **not one** `text`-labelled region has any of these
/// five as its whole read. The nearest legitimate neighbours are all on the other
/// side of a boundary this rule does not cross:
/// - the **comma-less** forms are ordinary dialogue and are excluded: `それは` is
///   11 rows on one region ("That would be..."), `そして` 7 rows on another ("And
///   then there's..."), `やっぱり` 7 rows on a third;
/// - the **longer** reads that merely begin with a member are excluded by the
///   whole-text requirement: two real `text` regions open on `そして、` and
///   `それでも、` and run on, and `それは．．．` is 12 rows on a real *effect*;
/// - a genuine effect that merely opens on the same character is untouched:
///   `しーいっ` (12 rows), `そう．．．` (13), `そっ` (9), `そんな` (7), and an
///   18-character exclamation opening on `その` (7).
///
/// So the band is not a numeric gap but a **partition**: five strings, zero
/// legitimate occurrences, and the closest thing to a collision differs by a
/// trailing `、` on a region carrying the other label.
///
/// **Why enumerated and not a register or "sounds like filler" lexicon.** This
/// project has a *measured* 50% false-positive rate on register lexicons: of 10
/// boilerplate-register hits in the OCR taxonomy, 5 were correct polite Japanese
/// -- a cinema usher, a chauffeur, a butler. A lexicon that generalises from
/// these five would be the same instrument.
///
/// **What is deliberately NOT in the set, though the same scan found it.** Eleven
/// further connective-shaped whole reads occur on onomatopoeia regions and are
/// all excluded, because none was hand-checked against the printed page and
/// several are plainly ordinary speech: `いや、` (8 rows, "No, wait,"), `よし、`
/// (7, "Alright,"), `やっぱり、` (7), `それじゃ、` (11), `ただただ、` (7),
/// `ひとつは、` (7), `彼女は、` (11), `まず、` (1), `これは今回は、`, `え、`,
/// `えー、`. `いや` and `よし` in particular are interjections a manga page says
/// constantly; adding them is exactly how the register lexicon failed. The rule
/// grows only by hand-checking a page, one string at a time.
const DRAWN_CONNECTIVES: &[&str] = &["それは、", "そして、", "しかし、", "そういえば、", "それでも、"];

/// Site marks burnt into pages served by comic aggregators: their stock
/// boilerplate (`最新免费漫画`, "latest free comics"; `本漫画由`, "this comic is
/// by"), plus Tencent Comics' own plate. Matched case-insensitively as
/// substrings. No aggregator's own brand is listed, so a bare brand name is not
/// refused. On the scoped arm (the default) a plate is still refused when, with
/// its site address and these marks removed, every remaining line contains a
/// [`SITE_PROSE`] word or is a near-miss misread of a mark or of a
/// [`SITE_PROSE`] word (see [`only_site_furniture`] and [`misread_plate`]).
const WATERMARKS: &[&str] = &[
    "最新免费漫画",
    // The traditional twin. Every other CJK entry here carries both variants and
    // these two did not, which matters more than it looks: `watermark_text` is a
    // literal `contains`, so a traditional read never reaches `story_text` at
    // all -- and `site_prose`, which would otherwise catch `漫畫`, is only
    // reachable from inside it. Prophylactic: no test corpus contains either
    // string, so neither claims a measured effect.
    "最新免費漫畫",
    "本漫畫由",
    "本漫画由",
    // Tencent. Found by a 60-page manhua sweep: one page lettered this one
    // "TENCENT COMICS" in black on a near-black plate on a black page, which is
    // unreadable, and erased the original glyphs to do it.
    "腾讯动漫",
    "騰訊動漫",
];

/// Site prose, kept BYTE-IDENTICAL to `koharu-pipeline`'s `stages/mod.rs` copy,
/// for the reason [`WATERMARKS`] itself is: the two answer one question at two
/// points, and a divergence lets a region be erased by one and lettered by the
/// other.
const SITE_PROSE: &[&str] = &[
    "漫畫", "漫画", "免費", "免费", "訪問", "访问", "收集整理",
];

/// One character folded for a case-insensitive compare.
fn lower_char(character: char) -> char {
    character.to_lowercase().next().unwrap_or(character)
}

/// Remove every occurrence of `mark`, comparing case-insensitively. Mirror of
/// the pipeline's `strip_mark`.
fn strip_mark(text: &str, mark: &str) -> String {
    let haystack: Vec<char> = text.chars().collect();
    let needle: Vec<char> = mark.chars().map(lower_char).collect();
    if needle.is_empty() || haystack.len() < needle.len() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < haystack.len() {
        let matches = index + needle.len() <= haystack.len()
            && (0..needle.len()).all(|step| lower_char(haystack[index + step]) == needle[step]);
        if matches {
            index += needle.len();
        } else {
            out.push(haystack[index]);
            index += 1;
        }
    }
    out
}

/// Whether the text carries a site mark at all. Mirror of the pipeline's
/// `watermark_text`. This used to be spelled out inline inside
/// [`only_site_furniture`]; it is a function now because [`story_text`] needs the
/// same question, and two spellings of one predicate is how the copies drift.
fn watermark_text(text: &str) -> bool {
    let lowered = text.to_lowercase();
    WATERMARKS
        .iter()
        .any(|mark| lowered.contains(&mark.to_lowercase()))
}

/// Fold fullwidth ASCII (U+FF01..U+FF5E) onto its halfwidth twin. Mirror of the
/// pipeline's `fold_fullwidth`.
///
/// PaddleOCR-VL returns a plate's address in fullwidth forms often enough to
/// matter. Every character is non-ASCII, so [`site_address`]'s trim eats the whole
/// token and the `.` test finds nothing to split on — the address is kept as
/// residue and the plate is lettered.
///
/// **MEASURED, and the first estimate was wrong by an order of magnitude.** Of the
/// 8 distinct fullwidth garbles across 93,588 regions in 16,918 run JSONs, exactly
/// **one** folds into a well-formed address (a misread site name with a `．ｃｏｎ`
/// tail), twice. The rest fail on no dot at all, a TLD too long, a non-alphabetic
/// TLD, or an empty head. **Claim two.** Folding [`orphaned_address_tail`] as well
/// was measured and buys nothing, so it is deliberately not done.
fn fold_fullwidth(token: &str) -> String {
    token
        .chars()
        .map(|character| match character as u32 {
            point @ 0xFF01..=0xFF5E => char::from_u32(point - 0xFEE0).unwrap_or(character),
            _ => character,
        })
        .collect()
}

/// A token shaped like a site address. Mirror of the pipeline's `site_address`.
fn site_address(token: &str) -> bool {
    let folded = fold_fullwidth(token);
    let token = folded.as_str();
    let trimmed = token.trim_matches(|character: char| !character.is_ascii_alphanumeric());
    if !trimmed
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '.' || character == '-')
    {
        return false;
    }
    trimmed.rsplit_once('.').is_some_and(|(head, tld)| {
        !head.is_empty()
            && (2..=4).contains(&tld.len())
            && tld.chars().all(|character| character.is_ascii_alphabetic())
    })
}

/// A domain's TAIL, orphaned when [`story_text`] removed the mark that was its
/// head. Mirror of the pipeline's `orphaned_address_tail`.
///
/// **This copy exists because the two must not disagree**, which is the reason
/// [`WATERMARKS`] is kept byte-identical: a divergence here would let a region be
/// erased by one gate and lettered by the other. The pipeline's copy carries the
/// measurement and the argument for trimming only the END.
fn orphaned_address_tail(token: &str) -> bool {
    let trimmed = token.trim_end_matches(|character: char| !character.is_ascii_alphanumeric());
    trimmed.strip_prefix('.').is_some_and(|tld| {
        (2..=4).contains(&tld.len()) && tld.chars().all(|character| character.is_ascii_alphabetic())
    })
}

/// Whether what is left of a line is still the site talking.
fn site_prose(line: &str) -> bool {
    let lowered = line.to_lowercase();
    SITE_PROSE.iter().any(|word| lowered.contains(word))
}

/// What a region says once the site's own text is removed. Mirror of the
/// pipeline's `story_text`; see that copy for why the scope is SUBSTRINGS and
/// not lines.
fn story_text(text: &str) -> String {
    let mut cleaned = text.to_string();
    for mark in WATERMARKS {
        cleaned = strip_mark(&cleaned, mark);
    }
    // GATED ON THE REGION ACTUALLY BEARING A MARK, and that gate is not tidiness.
    // `story_text` has TWO consumers and only one is the refusal: the pipeline's
    // `targets()` applies it to EVERY region's source before translation. Ungated,
    // the decoration rule below does not just tidy watermark residue -- it takes
    // punctuation off ordinary pages. Measured over 93,588 regions in 16,918 run
    // JSONs: ungated it changed the translator's effective input on 30
    // occurrences, 9 distinct, EVERY ONE non-watermark, a Korean bubble losing
    // its `!` among them. Gated: 0, refusal unchanged.
    let drop_decoration = watermark_text(text);
    cleaned
        .lines()
        .map(|line| {
            line.split_whitespace()
                .filter(|token| !site_address(token) && !orphaned_address_tail(token))
                .collect::<Vec<_>>()
                .join(" ")
        })
        // A line of pure DECORATION is not story text. The motivating plate is
        // `👇最新免费漫画👇`: with the mark stripped the two pointing hands
        // survived as residue, so `only_site_furniture`'s second half was false
        // and the banner was lettered onto the page as dialogue.
        //
        // `scripted()`, never `letters()`. `letters()` sums `other`, and
        // `is_symbol_or_punctuation` tops out at 0xFF64, so an emoji lands in
        // `other` and `letters("👇👇") == 2` -- a rule written on `letters` is a
        // no-op against the exact string this exists for.
        .filter(|line| {
            !line.trim().is_empty()
                && !site_prose(line)
                && !(drop_decoration && Scripts::of(line).scripted() == 0)
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Levenshtein distance in CHARACTERS, abandoned once it exceeds `cap`.
///
/// Characters and not bytes: every term this is asked about is CJK, where a byte
/// distance would count one misread glyph as three edits and no threshold could
/// mean anything.
fn edit_distance_within(a: &[char], b: &[char], cap: usize) -> Option<usize> {
    if a.len().abs_diff(b.len()) > cap {
        return None;
    }
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut current = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + usize::from(ca != cb));
        }
        previous = current;
    }
    (previous[b.len()] <= cap).then_some(previous[b.len()])
}

/// Whether what is left beside a site address is a MISREAD of a known site mark.
///
/// One character of a mark misread -- `漫晝` for `漫畫`, `本漫董由` for `本漫畫由` --
/// and a literal `contains` matches nothing, so the plate is translated and
/// lettered onto the page as though it were dialogue.
///
/// **A site address or a literal mark is the gate (see [`only_site_furniture`]),
/// and the misread match is only the confirmation**, which is what makes this
/// safe rather than a fuzzy matcher let loose on dialogue: a region with neither
/// never gets this far. A conjunction of two signals, deliberately: single
/// globally-tuned thresholds were tried for this class of rule and failed.
///
/// The tolerance scales with length, and that is load-bearing rather than tidy: a
/// flat "distance <= 2" would match an EMPTY residue against any two-character
/// term, so `www.paperlaf.test` would pass for the wrong reason and so would a
/// great deal else. Empty residue is handled by the address arm above it, never
/// here.
fn misread_plate(text: &str) -> bool {
    let residue: Vec<char> = story_text(text)
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    if residue.len() < 2 {
        return false;
    }
    for term in WATERMARKS.iter().chain(SITE_PROSE.iter()) {
        if term.is_ascii() {
            continue;
        }
        let term: Vec<char> = term.chars().collect();
        let cap = (term.len() / 2).clamp(1, 2);
        let width = term.len().min(residue.len());
        for start in 0..=residue.len() - width {
            if edit_distance_within(&residue[start..start + width], &term, cap).is_none() {
                continue;
            }
            /* THE MATCH IS NOT ENOUGH -- what is left once the misread mark is
             * taken out has to be furniture too, or this refusal would take a
             * skill name off the page with the plate that was stamped over it.
             * `本漫書由 Kaperleaf.test 碎星诀` is the shape of it: the mark is
             * misread AND the region carries a real name. Removing the window and
             * asking `scripted` leaves that region alone, at the price of also
             * sparing plates that happen to carry a caption. **That price is the
             * right way round** -- sparing a plate letters one wrong line,
             * refusing a name loses a real one. */
            let rest: String = residue[..start]
                .iter()
                .chain(residue[start + width..].iter())
                .collect();
            if Scripts::of(&rest).scripted() == 0 {
                return true;
            }
        }
    }
    false
}

/// Whether a region is nothing but site furniture. Mirror of the pipeline's
/// `only_site_furniture`, and the composed predicate `classify` calls.
///
/// **The two copies deliberately DIVERGE, by design: *refuse the lettering, keep
/// erasing*.** The pipeline's copy drives the erase mask (`withdraw_from_mask`);
/// this one drives lettering. Adding the misread-plate arm here and not there
/// means a misread plate is no longer lettered but is still erased -- which is
/// what the reader wants, since a cleanly erased plate is an improvement over a
/// plate left on the art. Do NOT "restore parity" by copying `misread_plate` into
/// `stages/mod.rs`: that would put the plate back on the artwork.
fn only_site_furniture(text: &str, scoped: bool) -> bool {
    let literal = watermark_text(text);
    if !scoped {
        return literal;
    }
    /* The mark may have been misread, so no literal can match it. Gate on the
     * address the recogniser DID get right, then confirm against the marks. */
    let addressed = text
        .split_whitespace()
        .any(|token| site_address(token) || orphaned_address_tail(token));
    if !literal && !addressed {
        return false;
    }
    /* BOTH arms end here, the literal one included.
     *
     * Returning `story_text(text).is_empty()` from the literal arm on its own
     * spares a plate whenever the recogniser read one mark RIGHT and another
     * WRONG: the correct mark makes `literal` true, the misread arm below is
     * never reached, and the garbled mark sits in `story_text` looking like
     * dialogue. Measured over the whole run archive, that shape was lettered
     * onto a page every time, while the plates whose marks were ALL garbled --
     * the harder case -- were refused, purely by accident of which part the
     * recogniser happened to miss.
     *
     * `misread_plate`'s own rescue is what makes this safe: it refuses only when
     * removing the matched mark leaves nothing SCRIPTED, so a plate stamped over
     * a skill name, or over a column of three names, keeps them and stays
     * lettered. Both are asserted below. */
    story_text(text).is_empty() || misread_plate(text)
}

/// The share of a region's letters that must be in the wrong script before it
/// counts as a bad read rather than a stray character. The same threshold the
/// offline segment labeller uses, deliberately, so the two stay comparable.
/// English function words, which carry no content and are never page text on a CJK
/// page.
///
/// **This closes the gap between the two Latin rules rather than moving either
/// threshold, and that is deliberate.** A description the decoder cut short lands as
/// `the` -- three letters, so [`Refusal::Junk`] (under three) does not catch it, and
/// one word, so the prose test (four or more) does not either. Two faint `THE`s
/// lettered across a test page's artwork for exactly that reason.
///
/// Widening `Junk` to three letters or narrowing the prose test to one word would
/// both be length rules, and this file's own record says what that costs: an
/// "under three characters" rule binned 62 Japanese segments of which 60 were real
/// dialogue. A closed set of function words cannot do that -- it can only ever match
/// a read that is entirely `the`, `a`, `of` and their like, which is not dialogue in
/// any language and is not a brand plate, a URL or a drawn effect either.
///
/// Kept SMALL on purpose. Every entry is a word that carries no content on its own;
/// `BOOM`, `AI`, `OK` and site names are not here and never will be.
const FUNCTION_WORDS: [&str; 18] = [
    "the", "a", "an", "of", "is", "are", "it", "its", "this", "that", "and", "or", "in",
    "on", "with", "to", "was", "were",
];

/// True when EVERY word is a function word. One content word anywhere spares the read.
fn only_function_words(text: &str) -> bool {
    let mut seen = false;
    for word in text.split_whitespace() {
        let word = word.trim_matches(|c: char| !c.is_alphanumeric()).to_ascii_lowercase();
        if word.is_empty() {
            continue;
        }
        seen = true;
        if !FUNCTION_WORDS.contains(&word.as_str()) {
            return false;
        }
    }
    seen
}

/// How many Latin words make a read PROSE rather than a plate.
///
/// Four, from the two populations it has to separate. The English descriptions
/// measured over a Chinese test chapter run 8 words and up ("The image contains
/// abstract pink and purple shapes"); the legitimate Latin on the same chapter is
/// a site address, one word each. There is a wide gap and four sits in it -- this
/// is not a threshold tuned to a boundary case.
const DESCRIPTION_WORDS: usize = 4;

const FOREIGN_SHARE: f64 = 0.34;

/// What the source language is, insofar as anything reliable says so.
///
/// **The pipeline does not detect this; it has to be told.** The per-region
/// language stamp in `stages/ocr.rs` (`stamped_language`) echoes the language the
/// request DECLARED, keeping `ja-JP` only for undeclared runs -- an echo of this
/// resolver's input, never detection, so a script rule reading it off the region
/// would be reading its own declaration back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceScript {
    Japanese,
    Korean,
    Chinese,
    /// Nothing trustworthy said. The script rule does not fire.
    Unknown,
}

impl SourceScript {
    /// Resolve from what the request declared, falling back to what the OCR
    /// engine implies.
    ///
    /// **The engine may only ever declare Japanese, never deny it.** `manga-ocr`
    /// is Japanese-only, so choosing it is positive evidence -- and the
    /// extension's per-host latch only picks it on a measured kana share
    /// (`background.js`, `SCRIPT_KANA_SHARE`). The converse is not evidence at
    /// all: PaddleOCR-VL reads Japanese perfectly well, so inferring "not
    /// Japanese" from "not manga-ocr" would fire the kana rule on genuine
    /// Japanese dialogue and
    /// silently drop it. That is the one failure this gate must not have, so an
    /// undeclared non-manga-ocr request resolves to `Unknown` and the four
    /// language-independent rules carry the page alone.
    #[must_use]
    pub fn resolve(declared: Option<&str>, ocr: Option<&str>) -> Self {
        if let Some(tag) = declared.map(str::trim).filter(|tag| !tag.is_empty()) {
            let tag = tag.to_ascii_lowercase();
            let primary = tag.split(['-', '_']).next().unwrap_or_default();
            return match primary {
                "ja" => Self::Japanese,
                "ko" => Self::Korean,
                "zh" | "cmn" | "yue" => Self::Chinese,
                _ => Self::Unknown,
            };
        }
        if ocr.is_some_and(|engine| engine.eq_ignore_ascii_case("manga-ocr")) {
            return Self::Japanese;
        }
        Self::Unknown
    }

    /// The same answer as a `Language`, for the one consumer downstream of the
    /// pipeline boundary: `PipelineConfig::translation.source_language`, which
    /// feeds the OCR stage's erase veto.
    ///
    /// **Not for the translator.** That field's own doc comment carries the full
    /// warning; the short version is that naming the source language in the
    /// prompt measured **-0.154 +/-0.071** over 511 segments, and
    /// `TranslationRequest::with_source_language` is how that arm was built.
    ///
    /// `Unknown` maps to `None` so the pipeline inherits the same
    /// nothing-fires-on-silence default this enum already has.
    #[must_use]
    pub fn language(self) -> Option<Language> {
        match self {
            Self::Japanese => Some(Language::Japanese),
            Self::Korean => Some(Language::Korean),
            Self::Chinese => "zh".parse().ok(),
            Self::Unknown => None,
        }
    }

    /// The resolved script as a wire tag, for the `format=json` body.
    ///
    /// **This exists because no other field records what the request resolved
    /// to.** The one that looks like it does — `regions[].source_language` —
    /// echoes the declared language but falls back to `ja-JP` when nothing was
    /// declared, so an auditor asking "what language was this arm run as?" cannot
    /// tell an undeclared run from a Japanese one. The two fields also differ in
    /// granularity: this tag is the bare script (`zh`), the region field a full
    /// BCP-47 tag (`zh-CN`) — so a string compare between them fails.
    ///
    /// `unknown` is spelled out rather than omitted, for the reason
    /// `TranslateJson::untranslated` gives: a present value means the server
    /// resolved and found nothing declared, an absent key would mean the server
    /// does not report it at all.
    ///
    /// Matches the `source_script=` field already in the tracing log, so the
    /// grep that finds one finds the other.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Japanese => "ja",
            Self::Korean => "ko",
            Self::Chinese => "zh",
            Self::Unknown => "unknown",
        }
    }
}

/// Why a region was refused. The strings match the offline segment labeller's
/// reasons so a rendered refusal and a benchmark label can be compared directly --
/// `DrawnConnective` is the one exception, because it is gated on the detector's
/// label and the Python labeller never saw one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Empty,
    Watermark,
    PunctuationOnly,
    /// A read consisting of nothing but `ー` and its halfwidth twin. Separate from
    /// [`Self::PunctuationOnly`] because the mark IS kana -- the script rules must
    /// keep seeing it as kana -- and because the reported reason is read by a human
    /// in the Diagnose panel, where "only punctuation" would be a small lie about
    /// what OCR actually returned.
    ProlongationOnly,
    Junk,
    ScriptMismatch,
    DrawnConnective,
    /// Not produced by this module: `duplicate::hide_duplicate_lettering` refuses
    /// the redundant half of a pair, and reports it through this enum so both
    /// gates reach the reader as one `refused` field.
    DuplicateLettering,
    /// No OCR engine was ever asked to read this region, so there is no verdict
    /// to report and never was one.
    ///
    /// **This is NOT [`Self::Empty`], and conflating them is the defect it
    /// exists to fix.** `Empty` means a `SourceText` component EXISTS holding an
    /// empty string -- an engine ran and came back with nothing. `Unread` means
    /// the component was never written at all, because `--skip-implausible-regions`
    /// dropped the target in `stages/ocr.rs` before it could become an OCR
    /// result. `regions.rs`'s `source: source.map(..).unwrap_or_default()`
    /// collapses both to `""` on the wire, which is precisely why the
    /// distinction has to be made here, where the `Option` is still alive.
    ///
    /// Produced by [`unread_regions`], not by [`hide_implausible`]: the walk
    /// that finds these must not be gated on the lettering flag, because whether
    /// an engine READ a region is independent of whether we would have lettered
    /// it.
    Unread,
    /// The engine DESCRIBED the picture instead of reading it.
    ///
    /// **Only a vision-LLM engine can produce this, and it is not hypothetical.**
    /// Over a 179-slice manhua chapter `minicpm-v-4.6` returned 20 English
    /// descriptions -- *"The image contains abstract pink and purple shapes..."* --
    /// and **all 20 were lettered**, one page carrying eleven such paragraphs
    /// across the artwork. `paddleocr-vl-1.6` cannot do this: it fabricates a short
    /// wrong word, which [`Self::Junk`] and the script rules already cover.
    ///
    /// **This is deliberately NOT a length rule, and the warning above about length
    /// still stands.** That warning is about binning SHORT reads: an "under three
    /// characters" rule binned 62 Japanese segments of which 60 were real dialogue.
    /// This rule can never fire on a short read. It fires only on Latin PROSE --
    /// several whitespace-separated words -- on a page whose declared script is
    /// CJK. Legitimate Latin on a manhua page is a URL, a brand plate or a short
    /// drawn effect (`www.paperleaf.test`, `BOOM`); it is not an English sentence.
    ImageDescription,
}

impl Refusal {
    #[must_use]
    pub const fn why(self) -> &'static str {
        match self {
            Self::Empty => "OCR returned nothing",
            Self::Watermark => "a site watermark, not dialogue",
            Self::PunctuationOnly => "no letters, only punctuation or symbols",
            Self::ProlongationOnly => "a prolongation mark with nothing to lengthen",
            Self::Junk => "one or two Latin letters",
            Self::ScriptMismatch => "the script does not belong to the source language",
            Self::DrawnConnective => "a dialogue connective read off a drawn sound effect",
            Self::DuplicateLettering => "the same text is lettered by an overlapping region",
            Self::Unread => "no OCR engine ever read it",
            Self::ImageDescription => "the engine described the picture instead of reading it",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Scripts {
    han: u32,
    kana: u32,
    hangul: u32,
    latin: u32,
    other: u32,
    /// Counted IN ADDITION to `kana`, never instead of it. `ー` really is kana for
    /// every script question -- a lone one on a Korean page is still a foreign
    /// mark -- so subtracting it here would quietly weaken `ScriptMismatch`. It is
    /// a second, narrower fact about the same characters.
    prolongation: u32,
}

impl Scripts {
    fn of(text: &str) -> Self {
        let mut counts = Self::default();
        for character in text.chars() {
            let point = character as u32;
            /* PUNCTUATION IS TESTED FIRST, and a green-looking test caught why:
             * `・` (U+30FB, katakana middle dot) lives INSIDE the kana block, so
             * a script test that reaches the block first counts a row of
             * interpuncts as six kana. On a Korean page that is 100% foreign
             * script and fires the mismatch rule -- the right refusal reported
             * for entirely the wrong reason, and on a Japanese page, where no
             * script rule fires at all, it is no refusal. Order, not ranges. */
            if is_symbol_or_punctuation(point)
                || character.is_whitespace()
                || character.is_ascii_punctuation()
                || character.is_ascii_digit()
            {
                continue;
            }
            if matches!(point, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF) {
                counts.han += 1;
            } else if matches!(point, 0x3040..=0x30FF | 0xFF66..=0xFF9F) {
                // The halfwidth katakana at FF66-FF9F are kana and are
                // `is_alphabetic`, so without them they would have been counted
                // as LATIN -- escaping the one rule they should trip.
                counts.kana += 1;
                if is_prolongation(point) {
                    counts.prolongation += 1;
                }
            } else if matches!(point, 0xAC00..=0xD7AF | 0x1100..=0x11FF) {
                counts.hangul += 1;
            } else if character.is_alphabetic() {
                counts.latin += 1;
            } else {
                counts.other += 1;
            }
        }
        counts
    }

    const fn letters(self) -> u32 {
        self.han + self.kana + self.hangul + self.latin + self.other
    }

    /// Letters in a *named* script. `other` is excluded on purpose: it is the
    /// bucket for things nobody classified, and dividing by it would move the
    /// threshold around for reasons unrelated to the script question.
    const fn scripted(self) -> u32 {
        self.han + self.kana + self.hangul + self.latin
    }
}

/// Ranges Unicode calls punctuation or symbol that `is_ascii_punctuation` misses:
/// CJK punctuation, fullwidth forms, arrows, geometric shapes, dingbats. A row of
/// stars or a lone `△` is a real beat in a real bubble, and it is also not
/// something anybody can translate.
const fn is_symbol_or_punctuation(point: u32) -> bool {
    matches!(point,
        0x30FB | 0xFF65   // katakana middle dot, full and half width
        // The LATIN middle dot, the mark above's Latin-1 twin. Missing, it fell
        // to the `other` bucket, `letters()` counted it, and a 16x16 detection
        // on a character's EYE that read `·` drew no refusal at all -- the iris
        // was erased and the interpunct lettered onto it. Byte-identical to
        // koharu-pipeline's copy; change both together.
        | 0x00B7
        | 0x309B | 0x309C // spacing voiced sound marks
        | 0x2000..=0x206F // general punctuation, including the ellipsis
        | 0x2190..=0x21FF // arrows
        | 0x2200..=0x22FF // mathematical operators
        | 0x2500..=0x257F // box drawing
        | 0x25A0..=0x25FF // geometric shapes
        | 0x2600..=0x27BF // misc symbols and dingbats
        | 0x3000..=0x303F // CJK symbols and punctuation
        | 0xFE30..=0xFE4F // CJK compatibility forms
        | 0xFF01..=0xFF20 // fullwidth ASCII punctuation and digits
        | 0xFF3B..=0xFF40
        | 0xFF5B..=0xFF64)
}

/// `ー` (U+30FC) and its halfwidth twin, which lengthen the vowel BEFORE them.
///
/// Kept byte-identical to `koharu-pipeline`'s copy in `stages/mod.rs`, for exactly
/// the reason `is_symbol_or_punctuation` above gives: the two answer the same
/// question at two points, and a divergence lets a region be erased by one and
/// lettered by the other. **That is not hypothetical here** -- shipping the
/// pipeline half alone left a webtoon test page with its artwork intact and the
/// em dash still drawn on it.
const fn is_prolongation(point: u32) -> bool {
    matches!(point, 0x30FC | 0xFF70)
}

/// A declared-KOREAN read carrying no hangul at all.
///
/// The ratio arm in `classify_scoped` cannot see this population: `scripted()`
/// includes latin, so an all-Latin misread of drawn hangul ("Vlok", "QIZE",
/// "KPV") divides to zero and `ka 山` to 1/3 -- both under `FOREIGN_SHARE`.
/// Korean is written in hangul; a read of `STRICT_HANGUL_MIN`+ scripted letters
/// with none is not a Korean read. The floor keeps one- and two-letter reads
/// with the rules that already own them (the junk rule; the ratio arm at 1/1).
///
/// A SEPARATE predicate rather than a new arm of `classify_scoped`, because it
/// rides its own flag (`--korean-script-strict`) and the walk composes it in
/// only when that flag and a declared-Korean script hold. Kept byte-identical
/// to `koharu-pipeline`'s `strict_korean_mismatch` -- the same contract
/// `is_symbol_or_punctuation` names: two answers to one question is a region
/// erased by one half and lettered by the other.
const STRICT_HANGUL_MIN: u32 = 3;

#[must_use]
pub fn strict_korean_mismatch(text: &str) -> bool {
    let counts = Scripts::of(text.trim());
    counts.hangul == 0 && counts.scripted() >= STRICT_HANGUL_MIN
}

/// Should this OCR string be lettered onto the page?
///
/// `None` means yes. The order matters only for which reason is reported.
#[must_use]
pub fn classify(text: &str, script: SourceScript) -> Option<Refusal> {
    classify_scoped(text, script, false)
}

/// [`classify`], with the watermark arm optionally scoped to the site's own text.
///
/// **Only the watermark arm is scoped, and that is deliberate.** Every other
/// refusal here is an aggregate over the whole string -- a count or a ratio -- and
/// is correctly whole-region. The watermark rule is the only positive SUBSTRING
/// match, which makes it the only one where one fragment's content can condemn
/// everything the detector legitimately drew around it.
///
/// `scoped == false` is `classify` op for op, so the shipped arm is unchanged.
pub fn classify_scoped(text: &str, script: SourceScript, scoped: bool) -> Option<Refusal> {
    let text = text.trim();
    if text.is_empty() {
        return Some(Refusal::Empty);
    }
    if only_site_furniture(text, scoped) {
        return Some(Refusal::Watermark);
    }

    let counts = Scripts::of(text);
    if counts.letters() == 0 {
        return Some(Refusal::PunctuationOnly);
    }
    /* A READ THAT IS NOTHING BUT PROLONGATION MARKS. `ー` lengthens the vowel
     * BEFORE it and has no sound of its own, so with nothing before it there is
     * no utterance -- and `Scripts::of` counts it as kana, so `letters()` above
     * is 1 and this needs its own line rather than a looser threshold.
     *
     * THIS IS THE HALF OF THE FIX THAT STOPS THE LETTERING. The pipeline's
     * `illegible_text` carries the same rule and withdraws the ERASE; measured on
     * a webtoon test page, that alone took the artwork destroyed from 8,535 px to
     * ZERO but still drew the em dash, because whether a region is lettered is
     * decided HERE and not there. Two copies, one question -- the hazard the
     * comment on `is_symbol_or_punctuation` names, met in the wild. Change both or
     * neither. */
    if counts.prolongation == counts.letters() {
        return Some(Refusal::ProlongationOnly);
    }
    if counts.latin > 0 && counts.letters() == counts.latin && counts.letters() < 3 {
        return Some(Refusal::Junk);
    }

    /* THE ENGINE DESCRIBED THE PICTURE INSTEAD OF READING IT.
     *
     * Sits beside `Junk` because both judge an all-Latin read, and apart from it
     * because they judge opposite ends: `Junk` catches one or two stray letters,
     * this catches an English sentence. Anything between the two is left alone.
     *
     * All three conditions are load-bearing:
     *   - every letter is Latin, so a read carrying ANY Han, kana or hangul is
     *     real page text and never reaches here;
     *   - the declared script is CJK. `Unknown` is excluded on purpose -- it means
     *     nothing trustworthy was said, and the other script rules decline to fire
     *     there for the same reason;
     *   - at least `DESCRIPTION_WORDS` whitespace-separated words, which is what
     *     separates prose from the Latin that legitimately appears on a manhua
     *     page: a URL, a brand plate, a short drawn effect.
     *
     * `www.paperleaf.test` is one word and survives. `纸叶漫画 paperleaf.test`
     * never reaches here at all, because it carries Han. */
    /* THE ENGINE'S OWN VOICE, IN CHINESE -- found from a render looked at rather
     * than a count: under `hunyuan-ocr-1.5` the falling-petals test page lettered
     * **"No text in image." eight times across the petals**, plus three "The
     * text in the image is: V/XW/d-o-T-o". HunyuanOCR
     * answers an artwork crop with a Chinese meta-sentence, which no rule here
     * could touch: it is Han, so the Latin-prose arm below never sees it, the
     * script arms agree with a `zh` page, and it is far over two letters.
     *
     * PREFIX-EXACT on the model's own formulas, deliberately not a "contains".
     * Greedy decode makes these stable verbatim, and a story sentence would have
     * to OPEN with "图片中没有/图片中的文本/图片中的文字" -- the narrator's voice
     * about an image it is inside -- to be lost. Unlike every scripted rule in
     * this chain it fires with NO declared language: the strings are the engine
     * talking about the picture in its own words, and the undeclared latch
     * window is exactly where the other description gates are dead.
     *
     * THE SAME VOICE WITH THE SHORTER LOCATIVE. `图中没有文字` walked past the
     * three formulas above on ONE missing character -- they open `图片中`, it
     * opens `图中` -- and lettered "There is no text in the image." across a
     * character's arm on one Chinese test page and "No text in image" onto
     * another's artwork, both looked at. `图中` is not a variant the model
     * invented; it is the wording of the QUESTION: `HUNYUAN_PROMPT` says
     * `请识别图中的所有文字` and contains `图片` nowhere, so a negative answer
     * echoing the question back lands on the shorter locative first. The
     * corpus prices the extension exactly: across 29,826 run JSONs, eighteen
     * distinct reads open with `图` and all eighteen are the engine; the one
     * genuine `图` on any page is mid-compound (a formation DIAGRAM).
     * `图中的文本` is gated although unobserved -- `文本` has ZERO
     * genuine occurrences anywhere in the corpus -- while `图中的文字` stays
     * deliberately ABSENT: `文字` is ordinary vocabulary and `图中的文字…` is
     * exactly how that diagram line would open. Byte-identical to
     * koharu-pipeline's copy (`stages/ocr.rs`), whose `strip_engine_preamble`
     * is the erase-protecting half of the same rule; the two move together or
     * neither does. */
    const ENGINE_META_PREFIXES: [&str; 5] = [
        "图片中没有",
        "图片中的文本",
        "图片中的文字",
        "图中没有",
        "图中的文本",
    ];
    if ENGINE_META_PREFIXES
        .iter()
        .any(|prefix| text.trim_start().starts_with(prefix))
    {
        return Some(Refusal::ImageDescription);
    }

    if counts.latin > 0
        && counts.letters() == counts.latin
        && matches!(
            script,
            SourceScript::Japanese | SourceScript::Korean | SourceScript::Chinese
        )
        && (text.split_whitespace().count() >= DESCRIPTION_WORDS || only_function_words(text))
    {
        return Some(Refusal::ImageDescription);
    }

    /* LENGTH IS NOT THE DISCRIMINATOR, and the Python side learned that the
     * expensive way: an "under three characters" rule binned 62 Japanese
     * segments of which 60 were real short dialogue -- interjections and beats,
     * which is most of what a manga page says -- while the genuinely bad reads
     * on the Korean and Chinese sources were kana and stray Han from LARGE
     * boxes, caught only by accident. Script is the test; there is no length
     * rule here and there should not be one. */
    let scripted = f64::from(counts.scripted());
    if scripted > 0.0 {
        match script {
            // Korean is written in Hangul. Han is possible in principle and
            // vanishingly rare in a modern webtoon; kana never belongs.
            SourceScript::Korean => {
                if f64::from(counts.han + counts.kana) / scripted >= FOREIGN_SHARE {
                    return Some(Refusal::ScriptMismatch);
                }
            }
            // Chinese has no kana. Han and Latin are both ordinary here.
            SourceScript::Chinese => {
                if f64::from(counts.kana) / scripted >= FOREIGN_SHARE {
                    return Some(Refusal::ScriptMismatch);
                }
            }
            // Japanese mixes kana and Han by design, so no rule fires on it --
            // and `Unknown` means nothing reliable said, so nothing fires there
            // either.
            SourceScript::Japanese | SourceScript::Unknown => {}
        }
    }
    None
}

/// Should this OCR string be lettered onto the page, given the **detector's own
/// class** for the region it was read from?
///
/// `None` means yes. Independent of [`classify`] and deliberately so: everything
/// in [`DRAWN_CONNECTIVES`] is perfectly ordinary Japanese, so no
/// language-general rule can or should refuse it. The only thing that makes it a
/// defect is being the whole read of a **drawn effect**, and that fact lives on
/// the region's label rather than in the string.
///
/// Two properties this signature buys, both load-bearing:
/// - `label` is passed rather than a `bool`, so a test can hand it the *same*
///   string under `Some("text")` and watch the rule stay silent. `そして、` in a
///   speech balloon is ordinary dialogue and must letter.
/// - the comparison is on the **whole** trimmed text, never a prefix. A real
///   effect may legitimately open on the same characters -- the corpus has
///   `それは．．．`, `そう．．．`, `そっ` and `しーいっ` on onomatopoeia regions --
///   and a longer sentence that merely begins with a connective is a sentence.
///
/// **It rides on `--letter-implausible-text` rather than taking a flag of its
/// own**, because it is refused through the same `hide_implausible` walk and the
/// same `Visibility` patch, so a separate switch would only let the two halves of
/// one gate disagree. `--letter-implausible-text` turns both off; there is no
/// arm that keeps the script rules and drops this one.
#[must_use]
pub fn classify_effect(text: &str, label: Option<&str>) -> Option<Refusal> {
    if label != Some(ONOMATOPOEIA) {
        return None;
    }
    DRAWN_CONNECTIVES
        .contains(&text.trim())
        .then_some(Refusal::DrawnConnective)
}

/// Hides every text layer whose OCR string [`classify`] refuses.
///
/// Returns the *content* entity of each refusal, which is the id `RegionOut`
/// carries, so the caller can report the reason on the region it belongs to.
/// Best-effort in the same way as `sfx::pin_sound_effects`: a page whose scene
/// will not take the patch still renders, just with the nonsense on it.
pub fn hide_implausible(
    session: &mut Session,
    page: EntityId,
    script: SourceScript,
    scope_watermark_refusals: bool,
    leave_misread_bubbles: bool,
    korean_script_strict: bool,
) -> Vec<(EntityId, Refusal)> {
    let snapshot = session.snapshot();
    let Ok(descendants) = snapshot.descendants(page) else {
        return Vec::new();
    };

    // Gathered by value before the patch opens, for the aliasing reason set out
    // in `lettering.rs`.
    let mut pending: Vec<(EntityId, EntityId, Refusal)> = Vec::new();
    for entity in descendants {
        if !matches!(entity.component::<TextLayout>(), Ok(Some(_))) {
            continue;
        }
        let Ok(layer) = snapshot.text_layer(entity.id()) else {
            continue;
        };
        let Ok(content) = layer.content() else { continue };

        /* FREE-STANDING TEXT ONLY, and the pixels are what put this here.
         *
         * The first build refused any region whose text failed the rules, and on
         * a dense Japanese page that emptied four balloons: the two countdown
         * bubbles reading `3` and `2`, and two carrying `!!` and an ellipsis.
         * Detection had already erased them to flat white, so refusing the
         * lettering left four blank holes -- visibly worse than the beat it was
         * suppressing. A balloon is a text CONTAINER: erase one and something
         * must go back in it. Artwork was never a container, so there the erase
         * is the whole of the damage and adding English to it only compounds it.
         *
         * `role` is the pipeline's own answer, written by `link_dialogue_regions`
         * for exactly the text it could place inside a bubble, and the corpus
         * says it separates the two populations almost perfectly: of 161 phantom
         * segments **147 are free-text and 14 are in-bubble**, against 194 usable
         * in-bubble. So this keeps 91% of the refusals and gives up every empty
         * balloon.
         *
         * `classify_effect` INHERITS this gate, and that is deliberate rather
         * than incidental. All 9 measured drawn-connective regions carry
         * `role: free-text`, so the gate costs the rule nothing on the population
         * it was derived from -- and an onomatopoeia detection that landed
         * *inside* a bubble is still a balloon somebody erased, where blanking
         * the lettering is the worse of the two defects. If such a region ever
         * turns up, it wants a page checked, not a bypass added here. */
        let free_standing = content
            .role()
            .ok()
            .flatten()
            .is_some_and(|value| !value.role.ends_with("dialogue"));

        let Ok(Some(source)) = content.source() else {
            continue;
        };

        /* The detector's own class for the region this text was recognised from,
         * read back exactly as `sfx::pin_sound_effects` reads it (`sfx.rs:123-131`)
         * -- `source_region` is the region, and its `Region.label` is the class.
         * Absent for a layer with no source region, which resolves to `None` and
         * therefore to no effect rule. */
        let label = content
            .source_region()
            .ok()
            .flatten()
            .and_then(|region| snapshot.component::<Region>(region.id()).ok().flatten())
            .and_then(|value| value.label);

        let Some(refusal) = classify_scoped(&source.text.value, script, scope_watermark_refusals)
            .or_else(|| classify_effect(&source.text.value, label.as_deref()))
            /* The strict Korean rule, composed in LAST so an existing rule's
             * reason wins the report when both fire. Scoped to a POSITIVELY
             * declared Korean script and its own flag; the pixel half is the same test
             * inside `withdraw_from_mask`'s strict arm. */
            .or_else(|| {
                (korean_script_strict
                    && matches!(script, SourceScript::Korean)
                    && strict_korean_mismatch(&source.text.value))
                .then_some(Refusal::ScriptMismatch)
            })
        else {
            continue;
        };

        /* THE FREE-STANDING GATE, APPLIED HERE RATHER THAN BEFORE THE CLASSIFY,
         * WITH ONE REFUSAL EXEMPT.
         *
         * The gate's reasoning above is measured and stands: blanking in-bubble
         * text leaves a HOLE, because detection erased the balloon to flat white
         * and a balloon is a container that something must go back into.
         *
         * `ProlongationOnly` is the one refusal for which that premise is FALSE.
         * The pipeline's `illegible_text` carries the byte-identical rule, so a
         * read this refuses is a read whose mask was already WITHDRAWN through
         * `write_illegible_veto` -- there is no flat white to leave behind, only
         * the original artwork. Measured on a webtoon test page: with the
         * pipeline half alone the artwork destroyed went 8,535 px -> 0 while the
         * em dash was still drawn, because this walk never reached the region.
         *
         * That is the whole reason the two copies of `is_prolongation` must stay
         * byte-identical -- the exemption is only sound while they agree. If they
         * ever diverge, this becomes the blank-hole defect the gate exists to
         * prevent. */
        /* With `--leave-misread-bubbles` on, a script or
         * punctuation refusal reaches DIALOGUE too -- for a POSITIVELY declared
         * zh/ko script only, so ja and Unknown keep today's behaviour exactly.
         *
         * The exemption is sound the same way ProlongationOnly's is: the mask
         * half moves with it. `withdraw_from_mask`'s script arm reads the SAME
         * flag (`MisreadLevers`), so a ScriptMismatch read's ink was never
         * erased; and a PunctuationOnly read is `letters() == 0`, whose pixel
         * twin is `illegible_text`'s unconditional `letters == 0` arm -- the two
         * counters skip the identical character set now that `0x00B7` is in
         * both punctuation tables. Refusing without withdrawing is the blank
         * hole this gate exists to prevent; the pairing is the lever. */
        let misread_exempt = leave_misread_bubbles
            && matches!(script, SourceScript::Korean | SourceScript::Chinese)
            && matches!(refusal, Refusal::ScriptMismatch | Refusal::PunctuationOnly);
        if !free_standing && refusal != Refusal::ProlongationOnly && !misread_exempt {
            continue;
        }
        pending.push((entity.id(), content.id(), refusal));
    }

    if pending.is_empty() {
        return Vec::new();
    }
    let hidden = snapshot.patch(|edit| {
        for (layer, _content, _refusal) in &pending {
            edit.set(
                *layer,
                &Visibility {
                    // Not `Generated`: no model produced this and no generation
                    // describes it. It is the operator's policy, applied to the
                    // finished scene, which is what `User` means here and what
                    // the other server-side passes use.
                    origin: Origin::User,
                    visible: false,
                    opacity: 1.0,
                },
            )?;
        }
        Ok(())
    });
    let Ok(patch) = hidden else { return Vec::new() };
    if session.commit(patch).is_err() {
        return Vec::new();
    }
    pending
        .into_iter()
        .map(|(_layer, content, refusal)| (content, refusal))
        .collect()
}

/// Every layer whose text content was never read by any OCR engine, as
/// `(content, Refusal::Unread)` -- the same shape [`hide_implausible`] returns,
/// so both merge into one `refused` field.
///
/// # Why this is a separate walk and not another arm of `hide_implausible`
///
/// Three reasons, and each one on its own is sufficient:
///
/// 1. **`hide_implausible` is gated and this must not be.** It runs only when
///    `state.skip_implausible_text && plan.letters_text()` (`routes.rs`). Whether
///    an engine *read* a region is decided by `--skip-implausible-regions` in the
///    pipeline's OCR stage, which is a different flag entirely; reporting the fact
///    must not depend on the lettering gate being on.
/// 2. **`hide_implausible` bails before it could see them.** Its walk needs a
///    `SourceText` to classify (`content.source()`, the `let ... else continue`
///    above) -- and the whole point of this population is that there is none.
/// 3. **It hides layers; this one must not.** These layers letter nothing already,
///    and the pipeline deliberately spares their artwork. This walk is
///    read-only: it takes a `&Snapshot` and commits nothing.
///
/// # The precondition, which is load-bearing
///
/// A region is reported here **only when it has neither a `SourceText` nor a
/// `Translation`.** That is not tidiness -- it is what keeps this change out of
/// two reader-facing channels. `regions::dropped_indices` requires a non-empty
/// `source` and `Stories::record` requires a non-empty source *and* translation,
/// so a region failing both was in neither list before and is in neither after.
/// Stamping a region that carries either one would silently change the
/// "N regions came back empty" bar, or drop a pair out of the story window that
/// the next page's prompt depends on.
///
/// # `Unread` is not `Empty`
///
/// `content.source()` returning `Ok(None)` means the component was never written.
/// `Ok(Some(text))` holding `""` means an engine ran and returned nothing, which
/// is [`Refusal::Empty`]'s job. Only the first is `Unread`. The wire cannot tell
/// them apart -- `regions.rs` maps both to `source: ""` -- so the distinction has
/// to be drawn here, while the `Option` still exists.
#[must_use]
pub fn unread_regions(snapshot: &Snapshot, page: EntityId) -> Vec<(EntityId, Refusal)> {
    let Ok(descendants) = snapshot.descendants(page) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entity in descendants {
        if !matches!(entity.component::<TextLayout>(), Ok(Some(_))) {
            continue;
        }
        let Ok(layer) = snapshot.text_layer(entity.id()) else {
            continue;
        };
        let Ok(content) = layer.content() else { continue };
        // `Ok(None)` -- never written -- is the whole population. `Ok(Some(_))`
        // is a read that happened, whatever it returned.
        if !matches!(content.source(), Ok(None)) {
            continue;
        }
        if !matches!(content.translation(), Ok(None)) {
            continue;
        }
        out.push((content.id(), Refusal::Unread));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use koharu_scene::{
        At, Authored, Geometry, PageDraft, RecognizedFrom, SourceText, TextLayoutKind, TextRegion,
        TextRole,
    };

    /// **The half of the prolongation fix that stops the LETTERING.**
    ///
    /// The pipeline's `illegible_text` withdraws the erase; this decides whether
    /// anything is drawn. Shipping only the first took the artwork destroyed on a
    /// webtoon test page from 8,535 px to zero and still drew the em dash, so this
    /// test exists because the render caught the gap, not because it was
    /// predicted.
    ///
    /// Asserted through `classify`, the function the region walk actually calls.
    #[test]
    fn a_read_of_only_prolongation_marks_is_not_lettered() {
        for script in [
            SourceScript::Japanese,
            SourceScript::Unknown,
            SourceScript::Korean,
            SourceScript::Chinese,
        ] {
            // What OCR returned off the character's own mouth line, verbatim.
            assert_eq!(classify("ー", script), Some(Refusal::ProlongationOnly));
            assert_eq!(classify("ｰ", script), Some(Refusal::ProlongationOnly));
            assert_eq!(classify("ーー", script), Some(Refusal::ProlongationOnly));
            // Trailing punctuation does not rescue it -- punctuation is skipped
            // before the count, so this is still nothing but the mark.
            assert_eq!(classify("ー…", script), Some(Refusal::ProlongationOnly));
        }
    }

    /// The reads that must still be lettered, each shaped like real output from
    /// the test corpora.
    ///
    /// `ん…` is the load-bearing case: a genuine balloon whose whole read is ONE
    /// character. It is why this rule is about the mark and not about length --
    /// and the reason the pages were rendered before the rule was written.
    #[test]
    fn ordinary_reads_carrying_a_prolongation_are_still_lettered() {
        for text in ["すごーい？", "それならー", "ん…", "あー", "冥霜之王！"] {
            assert_eq!(
                classify(text, SourceScript::Japanese),
                None,
                "{text:?} is a real read and must still be lettered"
            );
        }
        // The mark stays KANA for the script rules: a lone one on a Korean page is
        // still foreign, and counting it as prolongation must not weaken that.
        assert_eq!(
            classify("ソレデー", SourceScript::Korean),
            Some(Refusal::ScriptMismatch)
        );
    }

    #[test]
    fn resolves_the_declared_language_before_the_engine() {
        assert_eq!(
            SourceScript::resolve(Some("ko-KR"), Some("manga-ocr")),
            SourceScript::Korean
        );
        assert_eq!(
            SourceScript::resolve(Some("zh"), None),
            SourceScript::Chinese
        );
        assert_eq!(SourceScript::resolve(Some("  "), None), SourceScript::Unknown);
    }

    #[test]
    fn manga_ocr_declares_japanese_and_nothing_denies_it() {
        assert_eq!(
            SourceScript::resolve(None, Some("manga-ocr")),
            SourceScript::Japanese
        );
        // The load-bearing half: PaddleOCR-VL reads Japanese too, so it says
        // nothing about the source and must not enable the script rule.
        assert_eq!(
            SourceScript::resolve(None, Some("paddleocr-vl-1.6")),
            SourceScript::Unknown
        );
        assert_eq!(SourceScript::resolve(None, None), SourceScript::Unknown);
    }

    #[test]
    fn punctuation_and_symbols_are_not_text() {
        for text in ["......", "!!", "☆", "△", "・・・・・・", "、。", "  ...  "] {
            assert_eq!(
                classify(text, SourceScript::Unknown),
                Some(Refusal::PunctuationOnly),
                "{text}"
            );
        }
    }

    #[test]
    fn an_interpunct_is_punctuation_and_not_kana() {
        /* U+30FB sits inside the kana block. Counted as kana it makes a row of
         * interpuncts "100% foreign script" on a Korean page -- a refusal for
         * the wrong reason there, and none at all on a Japanese page. */
        for text in ["・・・", "・", "･･･"] {
            assert_eq!(
                classify(text, SourceScript::Korean),
                Some(Refusal::PunctuationOnly),
                "{text}"
            );
            assert_eq!(
                classify(text, SourceScript::Japanese),
                Some(Refusal::PunctuationOnly),
                "{text}"
            );
        }
        // The prolonged sound mark is NOT punctuation: it spells words.
        assert_eq!(classify("ラーメン", SourceScript::Japanese), None);
        assert_eq!(
            classify("ラーメン", SourceScript::Chinese),
            Some(Refusal::ScriptMismatch)
        );
    }

    #[test]
    fn halfwidth_katakana_counts_as_kana_rather_than_latin() {
        // `is_alphabetic` is true for these, so without their own range they
        // would be counted as Latin and escape the one rule they should trip.
        assert_eq!(
            classify("ﾋﾟﾝｸ", SourceScript::Chinese),
            Some(Refusal::ScriptMismatch)
        );
        assert_eq!(classify("ﾋﾟﾝｸ", SourceScript::Japanese), None);
    }

    #[test]
    fn one_or_two_latin_letters_are_a_detector_error() {
        assert_eq!(classify("O", SourceScript::Unknown), Some(Refusal::Junk));
        assert_eq!(classify("Y", SourceScript::Unknown), Some(Refusal::Junk));
        // Three is enough to be a word, and this is where the rule stops.
        assert_eq!(classify("Ack", SourceScript::Unknown), None);
    }

    #[test]
    fn short_japanese_dialogue_survives_every_rule() {
        /* The rule this guards is the one the Python side got wrong first, and
         * it cost 60 real segments. Every string here is two characters or
         * fewer and every one is ordinary speech. */
        for text in ["え", "うん", "はい", "あっ", "ね", "何"] {
            assert_eq!(classify(text, SourceScript::Japanese), None, "{text}");
            assert_eq!(classify(text, SourceScript::Unknown), None, "{text}");
        }
    }

    #[test]
    fn kana_on_a_chinese_page_is_refused_and_han_is_not() {
        // Read off a test page of falling petals with no text on it.
        for text in ["ピンク", "ピカ", "ひと", "ふっ"] {
            assert_eq!(
                classify(text, SourceScript::Chinese),
                Some(Refusal::ScriptMismatch),
                "{text}"
            );
        }
        // Ordinary Chinese must survive the same rule. This line is written for
        // the test rather than copied off a page.
        assert_eq!(classify("今天天气很好", SourceScript::Chinese), None);
        // ...and the same strings must survive when nobody said it was Chinese.
        assert_eq!(classify("ピンク", SourceScript::Unknown), None);
    }

    #[test]
    fn han_and_kana_on_a_korean_page_are_refused() {
        assert_eq!(
            classify("乾", SourceScript::Korean),
            Some(Refusal::ScriptMismatch)
        );
        assert_eq!(
            classify("ひと", SourceScript::Korean),
            Some(Refusal::ScriptMismatch)
        );
        assert_eq!(classify("안녕하세요", SourceScript::Korean), None);
    }

    #[test]
    fn a_stray_foreign_character_is_below_the_share() {
        // Nine Hangul and one Han: 10% foreign, far under the 34% threshold, so
        // the rule stays off. The threshold exists so a single OCR slip in a
        // real sentence is not a refusal.
        assert_eq!(classify("가나다라마바사아자乾", SourceScript::Korean), None);
    }

    #[test]
    fn watermarks_are_refused_in_any_language() {
        for script in [
            SourceScript::Japanese,
            SourceScript::Chinese,
            SourceScript::Korean,
            SourceScript::Unknown,
        ] {
            // A bare site address: refused on the scoped arm, which ships ON.
            assert_eq!(
                classify_scoped("www.paperleaf.test", script, true),
                Some(Refusal::Watermark)
            );
            assert_eq!(classify("最新免费漫画", script), Some(Refusal::Watermark));
            assert_eq!(
                classify("最新免费漫画 www.paperleaf.test", script),
                Some(Refusal::Watermark)
            );
        }
    }

    /// **A plate read CORRECTLY must still be refused.**
    ///
    /// A site name with its address, in any of the three shapes OCR returns it, is
    /// nothing but an address and [`SITE_PROSE`] once both are set aside.
    ///
    /// The orphan case is the subtle one. When a [`WATERMARKS`] entry is the head
    /// of a domain, stripping it leaves the orphan `.test`. [`site_address`] cannot
    /// see that: it trims the leading dot, finds `test`, and `rsplit_once('.')`
    /// returns `None`. Without [`orphaned_address_tail`] the residue is non-empty,
    /// so the region survives refusal, reaches the translator as the literal
    /// string `.test`, and is lettered.
    ///
    /// Asserted through [`classify_scoped`], which is what `hide_implausible`
    /// actually calls — not through `story_text` alone. The two halves of
    /// `only_site_furniture` are joined by an `&&`, and a test of the halves does
    /// not test the join.
    #[test]
    fn a_correctly_read_site_plate_is_refused_on_the_scoped_arm() {
        for plate in [
            "纸叶漫画 paperleaf.test",
            "纸叶漫画\npaperleaf.test",
            "纸叶漫画paperleaf.test",
            // A mark glued to the TLD: only the orphan rule refuses these.
            "最新免费漫画.test",
            "最新免费漫画\n最新免费漫画.test",
        ] {
            assert_eq!(
                classify_scoped(plate, SourceScript::Chinese, true),
                Some(Refusal::Watermark),
                "{plate:?} escaped the scoped refusal"
            );
        }
        // The control: a MISREAD address is refused too, because it survives
        // mark removal as a well-formed address. If this goes red the fix has
        // changed something it was not aimed at.
        assert_eq!(
            classify_scoped("纸叶漫画\npaperlesf.test", SourceScript::Chinese, true),
            Some(Refusal::Watermark)
        );
        // The control that matters more: the skill-name rescue must survive. A
        // skill name stamped with the plate is NOT furniture -- `碎星诀` here
        // stands for genuine story text.
        assert_eq!(
            classify_scoped("最新免费漫画\nKaperleaf.test\n碎星诀", SourceScript::Chinese, true),
            None
        );
        // And the unscoped arm is untouched: it never looked at the residue.
        assert_eq!(
            classify_scoped("最新免费漫画 www.paperleaf.test", SourceScript::Chinese, false),
            Some(Refusal::Watermark)
        );
    }

    /// **A plate carrying one mark read CORRECTLY and another misread is still a
    /// plate.**
    ///
    /// The correct mark makes `watermark_text` match. If the literal arm returned
    /// `story_text(..).is_empty()` on the spot, the garbled mark left in the
    /// residue would read as dialogue, while plates whose marks were ALL garbled
    /// -- the harder case -- were refused, so which part the recogniser missed
    /// would decide whether the reader saw the plate. Every string below is
    /// invented, in the shapes the run archive records.
    #[test]
    fn a_plate_whose_literal_mark_was_read_correctly_is_still_refused() {
        for plate in [
            "最新免费漫画\n本漫書由",   // 書 for 畫
            "最新免费漫画\n本漫蓄由",   // 蓄 for 畫
            "最新免费漫画 本漫  由",    // the mark truncated mid-word
            "最新免费漫画 本漫書由",
            "最新免费漫画\n收集整埋",   // 埋 for 理
        ] {
            assert_eq!(
                classify_scoped(plate, SourceScript::Chinese, true),
                Some(Refusal::Watermark),
                "{plate:?} was lettered onto a page and still escapes refusal"
            );
        }
    }

    /// **The other direction, and the one that decides the rule is safe to
    /// widen.** Every one of these carries a REAL name beside the plate, and all
    /// six must be spared. The third is the sharpest: three kings in a row,
    /// stamped with a plate whose address is garbled -- refusing it would take
    /// `苍炎之王`, `冥霜之王` and `裂风之王` off the page, and losing a skill name
    /// is the most serious defect class.
    #[test]
    fn widening_the_literal_arm_still_letters_every_name_bearing_plate() {
        for rescued in [
            "最新免费漫画\nKaperleaf.test\n碎星诀",
            "- 碧眼狼王-\n\n本漫書由\n\npaperleat.test",
            "苍炎之王·冥霜之王·裂风之王最新免费漫画 www.papcrleaf.test",
            "仓炎岛王王·冥雪岛王王\n最新免费漫画\nwww.paperlcaf.test",
            "- 碧眼狼王·\n\n本漫書由\n\npaperlaef.test",
            "紙葉漫畫  \n本漫畫由紙葉漫畫收集整理，更多免費漫畫請訪問  \nwww.paperlaf.test  pointing",
        ] {
            assert_eq!(
                classify_scoped(rescued, SourceScript::Chinese, true),
                None,
                "{rescued:?} carries story text and must survive the refusal"
            );
        }
    }

    /// **A plate whose mark the recogniser misread is still a plate.** One misread
    /// character defeats a literal `contains`; these are invented misreads of the
    /// listed marks, each beside a site address.
    ///
    /// `misread_plate` is joined to the address test by an `&&` and to the literal
    /// arm by the `if`, and a test of those pieces does not test the joins.
    #[test]
    fn a_plate_whose_brand_was_misread_is_still_refused() {
        for plate in [
            "漫晝 f.test",              // 晝 for 畫
            "最新兔费漫書\naperleaf.test", // 兔 for 免 and 書 for 画, and the p is gone
            "本漫董由\npaperleal.test",
            "纸页漫画 xaperleaf.test",   // site prose takes this one
            "本慢董由\npaperlcaf.test",  // TWO misreads
            "www.paperlaf.test",        // no mark at all; the address arm takes it
        ] {
            assert_eq!(
                classify_scoped(plate, SourceScript::Chinese, true),
                Some(Refusal::Watermark),
                "{plate:?} was lettered onto a page and still escapes refusal"
            );
        }
    }

    /// **THE CONTROL THAT DECIDES THE SHAPE OF THE RULE, and it is the skill-name
    /// rescue in its hardest form.**
    ///
    /// A skill name stamped with a plate is not furniture. The test above covers
    /// the case where the mark was read CORRECTLY. This is the case where the mark
    /// was misread AND a real name sits beside it — the shape that made the first
    /// draft of this rule wrong, because a bare near-mark match would have refused
    /// the whole region and taken the name off the page with the plate.
    ///
    /// So the rule removes the misread mark and asks whether anything SCRIPTED
    /// survives. The price is that a plate carrying a caption is spared and still
    /// letters one wrong line. **That price is the right way round**: sparing a
    /// plate letters one wrong line, refusing a name loses a real one.
    #[test]
    fn a_misread_plate_never_takes_a_skill_name_with_it() {
        for rescued in [
            "本漫書由\nKaperleaf.test\n碎星诀",          // misread mark + real name
            "- 碧眼狼王-\n\n本漫書由\n\npaperleat.test", // a name above the plate
        ] {
            assert_eq!(
                classify_scoped(rescued, SourceScript::Chinese, true),
                None,
                "{rescued:?} carries story text and must survive the refusal"
            );
        }
        // Ordinary dialogue is nowhere near this rule: no address, no gate.
        assert_eq!(
            classify_scoped("你今天的作业写完了吗？！", SourceScript::Chinese, true),
            None
        );
        // An address is the GATE, so text that merely resembles a mark and
        // carries no address is untouched. Without this the fuzzy match would be
        // loose on every page of every chapter.
        assert_eq!(classify_scoped("漫晝", SourceScript::Chinese, true), None);
        // And the unscoped arm never sees any of it.
        assert_eq!(classify_scoped("漫晝 f.test", SourceScript::Chinese, false), None);
    }

    /// **The refusal is defeated by DECORATION around a correctly-read plate.**
    ///
    /// The orphan-tail rule fixed the neighbouring case where the residue was an
    /// orphaned `.com` — 19 lettered watermarks to 1 over 140 pages. It does not
    /// reach this one. The motivating plate is `👇最新免费漫画👇`: the
    /// mark is stripped, the two pointing hands are not, [`story_text`] comes back
    /// as `👇👇`, and `only_site_furniture`'s second half is false. The banner is
    /// translated and lettered onto the page as though it were dialogue.
    ///
    /// The general rule is that a residue of PURE DECORATION is not story text.
    /// It must be tested with [`Scripts::scripted`] and never `letters`: `letters`
    /// sums `other`, and [`is_symbol_or_punctuation`] tops out at `0xFF64`, so an
    /// emoji lands in `other` and `letters("👇👇") == 2`. A rule written on
    /// `letters` is a no-op against the exact string this test exists for.
    ///
    /// The three decorations are deliberately different shapes: `👇` is astral and
    /// unclassified, `↓` is an arrow and `◇` a geometric shape, and the last two
    /// are already skipped by `Scripts::of` entirely. All three must reach the
    /// same verdict or the rule is keyed on the decoration rather than on the
    /// absence of script.
    #[test]
    fn a_decorated_site_plate_is_refused_on_the_scoped_arm() {
        for plate in [
            "\u{1F447}最新免费漫画\u{1F447}",
            "↓最新免费漫画↓",
            "◇最新免费漫画◇",
            "\u{1F447}最新免费漫画\u{1F447}\nwww.paperleaf.test",
        ] {
            assert_eq!(
                classify_scoped(plate, SourceScript::Chinese, true),
                Some(Refusal::Watermark),
                "{plate:?} escaped the scoped refusal"
            );
        }
        // THE RESCUE THAT MUST NOT MOVE. A skill name stamped with the plate is
        // not furniture, and `碎星诀` stands for real story text carrying real
        // script. If a decoration rule takes this it has stopped asking "is there
        // any script here" and started asking something else.
        assert_eq!(
            classify_scoped("最新免费漫画\nKaperleaf.test\n碎星诀", SourceScript::Chinese, true),
            None
        );
        // A decorated plate that ALSO carries story text keeps it, for the same
        // reason: the decoration is dropped, the sentence is not.
        assert_eq!(
            classify_scoped("\u{1F447}最新免费漫画\u{1F447}\n这是……！？", SourceScript::Chinese, true),
            None
        );
        // The unscoped arm never consults the residue, decorated or not.
        assert_eq!(
            classify_scoped("↓最新免费漫画↓", SourceScript::Chinese, false),
            Some(Refusal::Watermark)
        );
    }

    /// **A FULLWIDTH address is not recognised, so the plate survives.**
    ///
    /// PaddleOCR-VL returns the address in fullwidth forms often enough to matter.
    /// Every character is non-ASCII, so [`site_address`]'s trim eats the whole token
    /// and the `.` test then finds nothing to split on — the token is kept, the
    /// residue is non-empty, and the plate is lettered.
    ///
    /// **MEASURED, and the first estimate of this was wrong by an order of
    /// magnitude.** Of the 8 distinct fullwidth garbles across 93,588 regions in
    /// 16,918 run JSONs, exactly **one** folds into a well-formed address, twice.
    /// The rest fail on no dot at all, a TLD too long, a non-alphabetic TLD
    /// (`８ｍ`), or an empty head. **Claim two, not nineteen.** The plate below is
    /// invented in the shape of the one that folds.
    ///
    /// Folding [`orphaned_address_tail`] as well was measured and buys **nothing**,
    /// so it is deliberately not done.
    #[test]
    fn a_fullwidth_site_address_is_refused_on_the_scoped_arm() {
        assert_eq!(
            classify_scoped("最新免费漫画ｐａｐｅｒｌｅａｆ．ｔｅｓｔ", SourceScript::Chinese, true),
            Some(Refusal::Watermark),
            "the fullwidth address escaped the scoped refusal"
        );
        // The halfwidth twin was already refused and must stay so.
        assert_eq!(
            classify_scoped("最新免费漫画paperleaf.test", SourceScript::Chinese, true),
            Some(Refusal::Watermark)
        );
        // A fullwidth token that is NOT address-shaped is not swept up by the fold.
        // `８ｍ` folds to `8m`, which has no dot; `苍炎之王` is story text either way.
        assert_eq!(
            classify_scoped("最新免费漫画\n８ｍ\n苍炎之王", SourceScript::Chinese, true),
            None
        );
    }

    /// **The decoration rule must not reach a region with no watermark in it.**
    ///
    /// [`story_text`] has **two** consumers and only one of them is the refusal: the
    /// pipeline's `targets()` (`stages/translation.rs`) applies it to EVERY region's
    /// source text before translation. So an ungated decoration filter does not just
    /// tidy watermark residue — it takes punctuation off ordinary pages.
    ///
    /// **Measured over 93,588 regions in 16,918 run JSONs**: ungated, the rule
    /// changed the translator's effective input on **30 occurrences, 9 distinct,
    /// every one of them NON-watermark** — a Korean bubble losing its `!` among
    /// them. Gated, that
    /// number is **0**, with the refusal behaviour unchanged and a byte-identical
    /// control arm measuring 0 everywhere.
    ///
    /// Asserted on `story_text` directly rather than through `classify_scoped`,
    /// deliberately: the consumer at risk here is the translator, not the refusal,
    /// and `classify_scoped` cannot see a non-watermark region at all.
    #[test]
    fn decoration_is_only_dropped_from_a_watermark_region() {
        // Nothing here is a watermark, so nothing may be dropped. The first has
        // the shape of the corpus string that exposed this: a column of single
        // characters ending in `!`.
        assert_eq!(story_text("水\n나\n무\n돌\n요\n!"), "水\n나\n무\n돌\n요\n!");
        assert_eq!(story_text("1\nf"), "1\nf");
        assert_eq!(story_text("！？"), "！？");
        // With a watermark present the rule still fires — this is the arm the
        // decorated-plate tests above depend on.
        assert_eq!(story_text("\u{1F447}最新免费漫画\u{1F447}"), "");
        assert_eq!(
            story_text("\u{1F447}最新免费漫画\u{1F447}\n这是……！？"),
            "这是……！？"
        );
        assert_eq!(story_text("最新免费漫画\nKaperleaf.test\n碎星诀"), "碎星诀");
    }

    /// **A TRADITIONAL plate escapes the rule entirely.**
    ///
    /// [`WATERMARKS`] carries `最新免费漫画` and `腾讯动漫` in SIMPLIFIED form only,
    /// while every other CJK entry in it carries both variants. `watermark_text`
    /// is a literal `contains`, so a traditional read never even reaches
    /// [`story_text`] — and `site_prose`, which would otherwise catch `漫畫`, is
    /// only reachable from inside `story_text`. Nothing downstream catches it.
    ///
    /// No test corpus contains either traditional string, so this is
    /// prophylactic and claims no measured effect. It costs two array entries.
    #[test]
    fn a_traditional_site_plate_is_refused_on_the_scoped_arm() {
        for plate in ["最新免費漫畫", "騰訊動漫", "↓最新免費漫畫↓"] {
            assert_eq!(
                classify_scoped(plate, SourceScript::Chinese, true),
                Some(Refusal::Watermark),
                "{plate:?} escaped the scoped refusal"
            );
        }
    }

    #[test]
    fn empty_and_blank_are_refused() {
        assert_eq!(classify("", SourceScript::Unknown), Some(Refusal::Empty));
        assert_eq!(classify("   \n ", SourceScript::Unknown), Some(Refusal::Empty));
    }

    /// Every string here is the *whole* OCR read of a real onomatopoeia region on
    /// a Japanese test volume, and every one was checked against the printed page:
    /// the artwork carries no such text. Nine regions, five strings.
    #[test]
    fn a_connective_read_off_a_drawn_effect_is_refused() {
        for text in ["それは、", "そして、", "しかし、", "そういえば、", "それでも、"] {
            assert_eq!(
                classify_effect(text, Some("onomatopoeia")),
                Some(Refusal::DrawnConnective),
                "{text}"
            );
        }
        // Whitespace is trimmed, exactly as `classify` trims it.
        assert_eq!(
            classify_effect("  そして、  ", Some("onomatopoeia")),
            Some(Refusal::DrawnConnective)
        );
    }

    /// **The test that matters.** `そして、` in a speech balloon is perfectly
    /// ordinary dialogue, and the corpus says so: over 24,281 region rows, not
    /// one `text`-labelled region has any of the five as its whole read, while
    /// their comma-less cousins are real lines -- `それは` reads "That would
    /// be..." and `そして` "And then there's...". The label is the only thing
    /// that may fire this rule, so it is the only thing under test.
    #[test]
    fn the_same_connective_in_a_bubble_is_left_alone() {
        for text in ["それは、", "そして、", "しかし、", "そういえば、", "それでも、"] {
            assert_eq!(classify_effect(text, Some("text")), None, "text: {text}");
            // A layer with no source region, and a class nobody has seen, are
            // both "not an effect" rather than "probably an effect".
            assert_eq!(classify_effect(text, None), None, "none: {text}");
            assert_eq!(classify_effect(text, Some("bubble")), None, "bubble: {text}");
            assert_eq!(classify_effect(text, Some("panel")), None, "panel: {text}");
            // And the language-general rules must not refuse them either: they
            // are ordinary Japanese and `classify` is what runs on dialogue.
            assert_eq!(classify(text, SourceScript::Japanese), None, "{text}");
            assert_eq!(classify(text, SourceScript::Unknown), None, "{text}");
        }
    }

    /// The whole-text requirement, from both sides. A real effect may open on the
    /// same characters, and a real sentence may open with a real connective.
    #[test]
    fn a_genuine_effect_that_merely_starts_the_same_way_survives() {
        /* Read off onomatopoeia regions in stored runs: `それは．．．` 12 rows,
         * `そう．．．` 13, `しーいっ` 12, `そっ` 9, `それ` 9, `そんな` 7, and an
         * 18-character exclamation opening on `その` 7 (stood in for below by
         * an invented line of the same shape). None is in the set and none
         * may be refused, because a prefix rule would take the first four. */
        for text in [
            "それは．．．",
            "そう．．．",
            "しーいっ",
            "そっ",
            "それ",
            "そんな",
            "そのボタンだけは絶対に押さないで！！",
            "しかしっ",
            "そしてっ！！",
        ] {
            assert_eq!(
                classify_effect(text, Some("onomatopoeia")),
                None,
                "effect: {text}"
            );
        }
        /* The other direction: a real sentence beginning with a set member. Both
         * stand in, with invented text, for whole reads of real `text` regions
         * in stored runs that open the same way. */
        for text in [
            "そして、明日からの練習では、",
            "それでも、今日だけは",
        ] {
            assert_eq!(
                classify_effect(text, Some("onomatopoeia")),
                None,
                "sentence: {text}"
            );
            assert_eq!(classify_effect(text, Some("text")), None, "sentence: {text}");
        }
    }

    /// The connective-shaped reads the same corpus scan found and the set
    /// deliberately does NOT contain. `いや` and `よし` are interjections a manga
    /// page says constantly; none of these was hand-checked against a printed
    /// page, so none may fire. This is the register-lexicon trap written down as
    /// an assertion -- 5 of 10 boilerplate-register hits in the OCR taxonomy were
    /// correct polite Japanese, and a set that generalises is that instrument.
    #[test]
    fn the_set_does_not_generalise_to_every_connective_shaped_read() {
        for text in [
            "いや、",
            "よし、",
            "やっぱり、",
            "それじゃ、",
            "ただただ、",
            "ひとつは、",
            "彼女は、",
            "まず、",
            "え、",
            "えー、",
            "これは今回は、",
            // Not in any stored run -- the obvious next members of a lexicon
            // somebody would be tempted to write. They are not measured, so they
            // do not fire.
            "でも、",
            "だから、",
            "つまり、",
            "ところで、",
        ] {
            assert_eq!(
                classify_effect(text, Some("onomatopoeia")),
                None,
                "{text}"
            );
        }
    }

    /// A refusal that reports no reason is a refusal nobody can audit, and the
    /// two consequences (hidden from the render, dropped from the story window)
    /// are both keyed off this string reaching `RegionOut.refused`.
    #[test]
    fn the_drawn_connective_refusal_names_itself() {
        let why = Refusal::DrawnConnective.why();
        assert!(why.contains("connective"), "{why}");
        assert!(why.contains("sound effect"), "{why}");
        for other in [
            Refusal::Empty,
            Refusal::Watermark,
            Refusal::PunctuationOnly,
            Refusal::Junk,
            Refusal::ScriptMismatch,
        ] {
            assert_ne!(other.why(), why, "{other:?}");
        }
    }

    /// The exact reads that reached the page, verbatim from a `minicpm-v-4.6`
    /// chapter run, where all 20 such descriptions were LETTERED onto the artwork.
    #[test]
    fn an_english_description_of_the_picture_is_refused() {
        for text in [
            "The image contains abstract pink and purple shapes with some leaf-like forms.",
            "The image is predominantly pink with some white and dark purple areas.",
            "The image is too blurry to recognize any text.",
            "There are two prominent leaf-like forms outlined in pink.",
        ] {
            assert_eq!(
                classify(text, SourceScript::Chinese),
                Some(Refusal::ImageDescription),
                "{text}"
            );
        }
    }

    /// **The rule must not eat the Latin that legitimately appears on these pages.**
    /// Each is shaped like a real read from the same chapter: a site address, a
    /// short drawn effect, a chapter marker.
    #[test]
    fn the_latin_that_really_appears_on_a_manhua_page_survives() {
        for text in ["www.paperleaf.test", "paperleaf.test", "BOOM", "AI", "Ch.123"] {
            assert_ne!(
                classify(text, SourceScript::Chinese),
                Some(Refusal::ImageDescription),
                "{text}"
            );
        }
    }

    /// The exact reads that lettered on the falling-petals test page under
    /// `hunyuan-ocr-1.5` -- eight "No text in image." across the petals -- before
    /// this arm existed. Prefix-exact, and it needs NO
    /// declared script: the undeclared latch window is where the other
    /// description gates are dead, and the engine's own voice is engine-shaped
    /// in any language state.
    #[test]
    fn the_engines_chinese_meta_sentences_are_refused_in_every_language_state() {
        for text in [
            "图片中没有文字。",
            "图片中的文本内容是：V",
            "图片中的文本内容是：d",
        ] {
            for script in [SourceScript::Chinese, SourceScript::Unknown] {
                assert_eq!(
                    classify(text, script),
                    Some(Refusal::ImageDescription),
                    "{text} under {script:?}"
                );
            }
        }
    }

    /// **And the boundary of that arm**: dialogue that merely CONTAINS `图片`,
    /// or opens with something else, is page text and survives. Prefix-exact is
    /// the whole safety argument.
    #[test]
    fn dialogue_mentioning_a_picture_survives_the_meta_arm() {
        for text in ["这张图片中有什么？", "图片", "看图片中没有文字。"] {
            assert_ne!(
                classify(text, SourceScript::Chinese),
                Some(Refusal::ImageDescription),
                "{text}"
            );
        }
    }

    /// The SIBLING formula, on the locative the PROMPT itself uses. The first
    /// row is VERBATIM from two Chinese test chapters and from a `ja` page --
    /// note there is NO trailing `。`, so a rule keyed on the whole sentence
    /// would have missed it, and the `ja` exhibit is why the arm keeps its
    /// every-language-state scope. Rows three and four are DERIVED, not observed: they are what the
    /// prefix deliberately generalises over, and they go red if someone
    /// tightens it to the exact sentence.
    #[test]
    fn the_engines_shorter_locative_is_refused_in_every_language_state() {
        for text in [
            "图中没有文字",
            "图中没有文字。",
            "图中没有任何文字",
            "图中的文本内容是：叶",
        ] {
            for script in [
                SourceScript::Chinese,
                SourceScript::Japanese,
                SourceScript::Unknown,
            ] {
                assert_eq!(
                    classify(text, script),
                    Some(Refusal::ImageDescription),
                    "{text} under {script:?}"
                );
            }
        }
    }

    /// **The boundary of the shorter locative, from the corpus rather than from
    /// imagination.** Over 29,826 run JSONs exactly eighteen distinct reads
    /// open with `图` and all eighteen are the engine; the only genuine `图` on
    /// any page is mid-compound (a formation diagram; the first row here keeps
    /// an invented compound mid-sentence). The `无`-initial rows stand for real
    /// dialogue (14 rows over 5 distinct strings in the corpus), which is why no
    /// bare-negation opener may ever be gated. `图中的文字是山门旧训` is CONSTRUCTED in
    /// the register of the diagram line -- the one cross-product cell a real
    /// sentence can plausibly occupy, and the reason `图中的文字` is refused a place
    /// in the array while `图中的文本` has one. The four full-sentence rows are
    /// invented stand-ins of the same shape as the corpus rows they replace:
    /// the `图` compound stays mid-sentence and the `无` opener stays first.
    #[test]
    fn the_shorter_locative_arm_spares_real_page_text() {
        for text in [
            "这张地图是师父亲手画的草稿。",
            "无",
            "无处可逃",
            "无论你走到哪里，我都会在这条街的尽头等你回来。",
            "原创作品 没有",
            "看不清",
            "图中的文字是山门旧训",
            "被困在雨里的小猫，任何人都无法狠心走开",
            "灯火通明却无人回应",
            "看图中没有文字。",
        ] {
            assert_ne!(
                classify(text, SourceScript::Chinese),
                Some(Refusal::ImageDescription),
                "{text}"
            );
        }
    }

    /// A read carrying ANY CJK is page text, however long and however many spaces
    /// it has. This is the condition that keeps the LATIN-PROSE rule off ordinary
    /// dialogue. **One carve-out:** a read OPENING with the
    /// engine's own Chinese meta-formulas (`图片中没有`…) is the engine talking,
    /// not the page -- the test above pins it -- so "any Han" is now "any Han
    /// that is not the engine's own voice".
    #[test]
    fn a_read_with_any_han_is_never_a_description() {
        let text = "本漫畫由紙葉漫畫收集整理，更多免費漫畫請訪問 www.paperleaf.test";
        assert_ne!(
            classify(text, SourceScript::Chinese),
            Some(Refusal::ImageDescription)
        );
    }

    /// `Unknown` means nothing trustworthy was declared. Every other script rule
    /// declines to fire there and so does this one -- otherwise a page with no
    /// declared language would start refusing English.
    #[test]
    fn an_undeclared_script_never_fires_the_description_rule() {
        let text = "The image contains abstract pink and purple shapes.";
        assert_ne!(
            classify(text, SourceScript::Unknown),
            Some(Refusal::ImageDescription)
        );
    }

    /// **The boundary, asserted from both sides**, so a later edit to
    /// `DESCRIPTION_WORDS` cannot silently move it without a red test.
    /// The exact read that lettered two faint `THE`s across a test page's artwork: a
    /// description the decoder cut short, three letters and one word, falling
    /// between both Latin rules.
    #[test]
    fn a_truncated_description_left_as_a_function_word_is_refused() {
        for text in ["the", "The", "THE", "of the", "it is", "and"] {
            assert_eq!(
                classify(text, SourceScript::Chinese),
                Some(Refusal::ImageDescription),
                "{text}"
            );
        }
    }

    /// **One content word anywhere spares the read.** This is the condition that
    /// stops the function-word set behaving like a length rule.
    #[test]
    fn one_content_word_is_enough_to_survive_the_function_word_set() {
        for text in ["BOOM", "the BOOM", "a paperleaf", "OK", "AI", "is it BOOM"] {
            assert_ne!(
                classify(text, SourceScript::Chinese),
                Some(Refusal::ImageDescription),
                "{text}"
            );
        }
    }

    #[test]
    fn the_word_floor_is_asserted_from_both_sides() {
        assert_ne!(
            classify("one two three", SourceScript::Chinese),
            Some(Refusal::ImageDescription),
            "three words is a plate, not prose"
        );
        assert_eq!(
            classify("one two three four", SourceScript::Chinese),
            Some(Refusal::ImageDescription),
            "four words is the floor"
        );
    }

    #[test]
    fn ordinary_dialogue_is_never_refused() {
        for (text, script) in [
            ("Hello there", SourceScript::Unknown),
            ("明日は晴れるといいな", SourceScript::Japanese),
            ("你来晚了！我先走！", SourceScript::Chinese),
            ("이건 정말로 따뜻한", SourceScript::Korean),
        ] {
            assert_eq!(classify(text, script), None, "{text}");
        }
    }

    /// One page carrying one free-standing text layer, built in the exact shape
    /// `hide_implausible` walks: a `TextLayout` layer presenting a `TextContent`
    /// that carries `SourceText` and `TextRole`, and is `RecognizedFrom` an
    /// analysis region whose `Region.label` is the detector's class.
    ///
    /// The region's geometry is the box the measured drawn-connective defect
    /// occupies -- (178.9, 742.6, 239.7, 230.4) on an 844x1200 page -- so the
    /// fixture is the measured defect rather than an invented one.
    ///
    /// Returns the session, the page, and the layer, so a caller can ask the
    /// scene itself whether the layer was hidden rather than trusting the return
    /// value alone.
    fn page_with_one_text(
        text: &str,
        label: Option<&str>,
        role: &str,
    ) -> (Session, EntityId, EntityId) {
        let mut session = Session::memory().expect("an in-memory session");
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 844.0, 1200.0), At::End)?;
                let region = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(178.9, 742.6, 239.7, 230.4),
                    label.map(str::to_owned),
                )?;
                let content = edit.add_text_content(page, At::End)?;
                edit.set(
                    content,
                    &SourceText {
                        text: Authored::user(text.to_owned()),
                        language: None,
                    },
                )?;
                edit.set(
                    content,
                    &TextRole {
                        origin: Origin::User,
                        role: role.to_owned(),
                    },
                )?;
                let layer = edit.add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: Origin::User,
                        kind: TextLayoutKind::Paragraph,
                    },
                )?;
                edit.relate::<RecognizedFrom>(content, region)?;
                ids = Some((page, layer));
                Ok(())
            })
            .expect("the fixture scene is valid");
        session.commit(patch).expect("the fixture scene commits");
        let (page, layer) = ids.expect("the edit ran to completion");
        (session, page, layer)
    }

    /// A page carrying the three OCR outcomes side by side, returned as content
    /// ids in that order: **never read**, **read and blank**, **read normally**.
    ///
    /// The middle one is the whole reason this fixture exists. On the wire it is
    /// indistinguishable from the first -- `regions.rs` maps both to `source: ""`
    /// -- so a test built from only the first and third would pass with the two
    /// causes conflated, which is the defect this distinction exists to prevent.
    fn page_with_three_read_states() -> (Session, EntityId, [EntityId; 3]) {
        let mut session = Session::memory().expect("an in-memory session");
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 844.0, 1200.0), At::End)?;
                let mut contents = Vec::new();
                // Each content needs its OWN region: `regions::regions` reads the
                // box off the recognized-from region and skips any layer it cannot
                // find geometry for, so sharing one would drop two of the three.
                for (index, text) in [None, Some(""), Some("ふきだし")].into_iter().enumerate() {
                    let region = edit.add_analysis_region::<TextRegion>(
                        page,
                        At::End,
                        &Geometry::rectangle(10.0 + 300.0 * index as f64, 20.0, 239.7, 230.4),
                        Some("onomatopoeia".to_owned()),
                    )?;
                    let content = edit.add_text_content(page, At::End)?;
                    // `None` writes NO component at all -- not an empty one. That
                    // is the distinction the walk turns on.
                    if let Some(text) = text {
                        edit.set(
                            content,
                            &SourceText {
                                text: Authored::user(text.to_owned()),
                                language: None,
                            },
                        )?;
                    }
                    edit.set(
                        content,
                        &TextRole {
                            origin: Origin::User,
                            role: "dev.koharu.text.free-text".to_owned(),
                        },
                    )?;
                    edit.add_text_layer(
                        page,
                        At::End,
                        content,
                        &TextLayout {
                            origin: Origin::User,
                            kind: TextLayoutKind::Paragraph,
                        },
                    )?;
                    edit.relate::<RecognizedFrom>(content, region)?;
                    contents.push(content);
                }
                ids = Some((page, [contents[0], contents[1], contents[2]]));
                Ok(())
            })
            .expect("the fixture scene is valid");
        session.commit(patch).expect("the fixture scene commits");
        let (page, contents) = ids.expect("the edit ran to completion");
        (session, page, contents)
    }

    /// The unread/empty distinction, asserted on **what the caller actually calls**.
    ///
    /// The chain is scene → `regions::regions` → [`unread_regions`] →
    /// `regions::stamp_refusals` → `RegionOut.refused`, and it is driven end to
    /// end here because asserting on the walk alone would stay green with the
    /// join deleted. That is not hypothetical: a fix that nothing calls passes
    /// every test of the fix.
    ///
    /// The `&&` inside the walk is what the middle region tests. A predicate that
    /// only asked "is `source` empty?" would refuse a region an engine really did
    /// read, and would report `"no OCR engine ever read it"` about a page where
    /// one did.
    #[test]
    fn a_region_no_engine_read_is_named_on_the_wire_and_a_blank_read_is_not() {
        let (session, page, [never_read, read_blank, read_ok]) = page_with_three_read_states();
        let snapshot = session.snapshot();

        let unread = unread_regions(&snapshot, page);
        assert_eq!(
            unread,
            vec![(never_read, Refusal::Unread)],
            "only the content with no SourceText at all is unread"
        );

        let mut regions = crate::regions::regions(&snapshot, page);
        assert_eq!(regions.len(), 3, "the fixture must reach the wire: {regions:?}");
        crate::regions::stamp_refusals(&mut regions, &unread);

        let refused_for = |content: EntityId| {
            regions
                .iter()
                .find(|region| region.content == content)
                .unwrap_or_else(|| panic!("{content} is missing from regions[]"))
                .refused
        };
        assert_eq!(
            refused_for(never_read),
            Some("no OCR engine ever read it"),
            "the region no engine read must be NAMED, not silent"
        );
        assert_eq!(
            refused_for(read_blank),
            None,
            "an engine ran and returned nothing -- that is Refusal::Empty's business, not this one"
        );
        assert_eq!(refused_for(read_ok), None, "a normal read is untouched");

        // The precondition that keeps this out of `dropped` and the story window:
        // both of those require a non-empty source, and the stamped region has
        // none. If this ever fails, the reader-facing "N regions came back empty"
        // bar and the next page's prompt are both in scope again.
        let stamped = regions
            .iter()
            .find(|region| region.content == never_read)
            .expect("the stamped region");
        assert!(
            stamped.source.is_empty() && stamped.translated.is_empty(),
            "Unread may only be stamped on a region with neither source nor translation"
        );
    }

    /// The two causes must not collapse into one string. `Empty` says an engine
    /// ran and came back with nothing; `Unread` says none was ever asked.
    #[test]
    fn unread_and_empty_are_different_reasons() {
        assert_ne!(Refusal::Unread.why(), Refusal::Empty.why());
        assert_eq!(Refusal::Unread.why(), "no OCR engine ever read it");
        assert_eq!(Refusal::Empty.why(), "OCR returned nothing");
    }

    /// `false` when the layer carries no `Visibility` at all, which is what an
    /// unrefused layer looks like: `hide_implausible` is the only thing in this
    /// module that writes one.
    fn is_hidden(session: &Session, layer: EntityId) -> bool {
        session
            .snapshot()
            .component::<Visibility>(layer)
            .expect("the layer is in the scene")
            .is_some_and(|visibility| !visibility.visible)
    }

    /// **The test the pure-function tests could not be.**
    /// `the_same_connective_in_a_bubble_is_left_alone` exercises
    /// [`classify_effect`] directly, so it stays green however the caller reads
    /// the label -- replacing the whole `source_region().. label` block in
    /// [`hide_implausible`] with an unconditional `Some("onomatopoeia")` left the
    /// entire suite passing. That is the classic failure mode: the rule was
    /// tested and the *wiring* was not, and the wiring is the half that can
    /// fire on every region on the page.
    ///
    /// So this drives `hide_implausible` itself, over a real scene, and asks the
    /// same string under two different labels.
    #[test]
    fn the_label_that_fires_the_rule_is_read_off_the_source_region() {
        // The population the rule must never touch. `そして、` in a balloon is
        // ordinary dialogue, and a caller that stopped reading the label would
        // refuse it.
        let (mut session, page, layer) =
            page_with_one_text("そして、", Some("text"), "dev.koharu.text.free-text");
        assert_eq!(
            hide_implausible(&mut session, page, SourceScript::Japanese, false, false, false),
            Vec::new(),
            "a `text` region must not be refused"
        );
        assert!(!is_hidden(&session, layer), "the layer must still render");

        // ...and the same string on a drawn effect is the measured defect.
        let (mut session, page, layer) =
            page_with_one_text("そして、", Some("onomatopoeia"), "dev.koharu.text.free-text");
        let refused = hide_implausible(&mut session, page, SourceScript::Japanese, false, false, false);
        assert_eq!(refused.len(), 1, "{refused:?}");
        assert_eq!(refused[0].1, Refusal::DrawnConnective);
        assert!(is_hidden(&session, layer), "the layer must be hidden");
    }

    /// A region the detector gave no class to is "not an effect", never
    /// "probably an effect" -- and the `and_then(|value| value.label)` that makes
    /// that true is the same line as above.
    #[test]
    fn an_unlabelled_region_never_fires_the_effect_rule() {
        let (mut session, page, layer) =
            page_with_one_text("そして、", None, "dev.koharu.text.free-text");
        assert_eq!(
            hide_implausible(&mut session, page, SourceScript::Japanese, false, false, false),
            Vec::new()
        );
        assert!(!is_hidden(&session, layer));
    }

    /// The free-standing gate, from the caller's side. An erased balloon with
    /// nothing put back is the worse of the two defects, so an `onomatopoeia`
    /// detection that landed inside a bubble is left alone even though
    /// [`classify_effect`] would refuse the string.
    #[test]
    fn a_connective_on_a_dialogue_layer_is_left_alone() {
        assert_eq!(
            classify_effect("そして、", Some("onomatopoeia")),
            Some(Refusal::DrawnConnective),
            "the rule itself must still refuse it, or this proves nothing"
        );
        let (mut session, page, layer) =
            page_with_one_text("そして、", Some("onomatopoeia"), "dev.koharu.text.dialogue");
        assert_eq!(
            hide_implausible(&mut session, page, SourceScript::Japanese, false, false, false),
            Vec::new()
        );
        assert!(!is_hidden(&session, layer));
    }

    /// The language-general half of the same walk, so a break in the shared
    /// plumbing cannot be mistaken for a break in the effect rule.
    #[test]
    fn the_script_rules_reach_the_scene_too() {
        let (mut session, page, layer) =
            page_with_one_text("ピンク", Some("text"), "dev.koharu.text.free-text");
        let refused = hide_implausible(&mut session, page, SourceScript::Chinese, false, false, false);
        assert_eq!(refused.len(), 1, "{refused:?}");
        assert_eq!(refused[0].1, Refusal::ScriptMismatch);
        assert!(is_hidden(&session, layer));
    }

    /// The strict Korean predicate: a declared-Korean read with no hangul at
    /// three or more scripted letters. Fire and not-fire on invented strings in
    /// the shapes of the measured exhibits.
    #[test]
    fn strict_korean_mismatch_fires_on_the_leaked_population_and_nothing_real() {
        // All-Latin leaks and a mixed one, in the measured shapes.
        for leak in ["Vlok", "QIZE", "KPV", "ZUMB", "ka 山"] {
            assert!(strict_korean_mismatch(leak), "{leak}");
        }
        // Real Korean, a hangul-bearing mix, and the populations other rules
        // already own: the junk rule (1-2 latin) and the ratio arm (1/1 han).
        for real in ["안녕하세요", "흠…", "OK!", "我", "yo 안"] {
            assert!(!strict_korean_mismatch(real), "{real}");
        }
    }

    /// The strict arm reaches the walk only under its flag AND a declared
    /// Korean script -- and through the same free-standing gate as every other
    /// refusal, so a dialogue leak needs BOTH levers.
    #[test]
    fn the_strict_arm_is_scoped_to_its_flag_and_the_declared_script() {
        let run = |script, leave, strict| {
            let (mut session, page, _layer) =
                page_with_one_text("Vlok", Some("text"), "dev.koharu.text.free-text");
            hide_implausible(&mut session, page, script, false, leave, strict)
        };
        assert_eq!(run(SourceScript::Korean, false, false), Vec::new(), "off arm");
        assert_eq!(
            run(SourceScript::Korean, false, true).first().map(|r| r.1),
            Some(Refusal::ScriptMismatch),
            "free-standing Vlok on declared ko refuses under the strict flag alone"
        );
        assert_eq!(run(SourceScript::Chinese, false, true), Vec::new(), "zh is out of scope");
        assert_eq!(run(SourceScript::Unknown, false, true), Vec::new(), "undeclared fires nothing");
    }

    /// The `我` split, resolved. The same single-Han read on
    /// a declared-Korean page was refused free-standing and lettered in a
    /// bubble; with the lever on, the bubble is refused too -- and the mask half
    /// rides the same flag pipeline-side, which is what makes this sound.
    #[test]
    fn a_misread_bubble_is_refused_only_when_the_lever_asks() {
        let run = |leave| {
            let (mut session, page, layer) =
                page_with_one_text("我", Some("text"), "dev.koharu.text.dialogue");
            let refused = hide_implausible(&mut session, page, SourceScript::Korean, false, leave, false);
            (refused, is_hidden(&session, layer))
        };
        let (off, hidden_off) = run(false);
        assert_eq!(off, Vec::new(), "today's behaviour: the dialogue gate protects it");
        assert!(!hidden_off);
        let (on, hidden_on) = run(true);
        assert_eq!(on.first().map(|r| r.1), Some(Refusal::ScriptMismatch), "{on:?}");
        assert!(hidden_on, "the wrong word must not letter");
    }

    /// The pupil, end to end on the walk: `·` classifies as PunctuationOnly now
    /// that U+00B7 is punctuation, and the lever lets that refusal reach the
    /// dialogue role the eye was linked under -- on declared zh/ko only.
    #[test]
    fn the_interpunct_is_punctuation_and_the_lever_reaches_the_pupil() {
        assert_eq!(
            classify("·", SourceScript::Korean),
            Some(Refusal::PunctuationOnly),
            "U+00B7 must count as punctuation, not as a letter"
        );
        let run = |script, leave| {
            let (mut session, page, _layer) =
                page_with_one_text("·", Some("text"), "dev.koharu.text.dialogue");
            hide_implausible(&mut session, page, script, false, leave, false)
        };
        assert_eq!(run(SourceScript::Korean, false), Vec::new(), "gated today");
        assert_eq!(
            run(SourceScript::Korean, true).first().map(|r| r.1),
            Some(Refusal::PunctuationOnly),
            "the lever frees the refusal"
        );
        assert_eq!(
            run(SourceScript::Japanese, true),
            Vec::new(),
            "ja keeps today's behaviour exactly -- the lever is scoped to declared zh/ko"
        );
    }

    /// Real Korean dialogue is untouchable under every lever combination --
    /// the one regression this lever must not have.
    #[test]
    fn real_korean_dialogue_survives_every_lever_combination() {
        for (leave, strict) in [(false, false), (true, false), (false, true), (true, true)] {
            let (mut session, page, layer) =
                page_with_one_text("안녕하세요", Some("text"), "dev.koharu.text.dialogue");
            assert_eq!(
                hide_implausible(&mut session, page, SourceScript::Korean, false, leave, strict),
                Vec::new(),
                "leave={leave} strict={strict}"
            );
            assert!(!is_hidden(&session, layer));
        }
    }
}
