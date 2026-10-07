/* BireLate - which STORY a page belongs to.
 *
 * A story is the translation-context window the server carries from page to
 * page: `server/src/story.rs` keeps the last 96 source/target pairs under a
 * string id, and every request naming that id is prompted with them. That window
 * is what stops a surname being romanised three different ways across a volume.
 *
 * UNTIL THIS FILE EXISTED THE WINDOW WAS SCOPED TO THE BROWSER PROFILE.
 * `background.js` minted one UUID the first time anything was translated and
 * stored it in `storage.local` for ever, so a reader who finished one volume and
 * opened a different title carried the first title's names into the second as
 * established precedent -- and the only way out was a button in the popup.
 *
 * The gap was visible in the justification for the feature. The measurement that
 * put a story on by default was taken over 40 pages of ONE VOLUME; it is
 * evidence for a per-WORK window and was implementing a per-PROFILE one.
 *
 * WHAT THIS FILE IS. The URL -> story-key rule and nothing else: no DOM, no
 * fetch, no storage, no browser API at all. Same shape as seam.js, for the same
 * reason -- it is a CLASSIC script sharing one global with cache.js and
 * background.js, and `node --test tests/story-key.test.js` loads it directly
 * through the `typeof module` guard at the bottom, which is the only harness
 * this rule has.
 *
 * Every top-level name here starts with `story` or `STORY_`, EXCEPT that
 * `storyId` is deliberately avoided: `background.js` already declares a function
 * of that name at top level, and one name declared twice across two classic
 * scripts sharing a global is a SyntaxError that kills both. Hence `storyIdFor`.
 *
 * ---------------------------------------------------------------------------
 * THE SCOPE IS A SERIES, NOT A CHAPTER. This is the decision the rule below is
 * built to serve and it is worth stating before the regexes, because half of
 * them exist to THROW AWAY the chapter. The measured benefit is name consistency
 * across the pages of a volume; a chapter boundary would reset the window at
 * exactly the moment a romanisation has finally settled.
 *
 * THE URL SHAPES this rule was designed against. Site names, slugs and ids are
 * invented examples; what matters is the shape -- path depth, chapter tokens,
 * the query parameter:
 *
 *   paperleaf  /comic/chapter/sample-saga_abcxyz/0_207.html
 *              ...and /0_207_2.html, because one chapter is FOUR html pages
 *   stripline  /webs/comic-next/381206   and   /webs/comic-next/382913
 *              -- chapters 1 and 2 of ONE series. The series
 *              appears NOWHERE in the reader URL. See the fallback below.
 *   portal     /webtoon/detail?titleId=700001&no=163&week=thu
 *              -- the series is a QUERY parameter; `no` is the episode.
 *   scanhouse  /read/iron-monarch/ch1-52817/
 *   tinyengine /tiny-engine-chapter-163/     (w27.readtinyengine.test)
 *   quietgod   /the-quiet-god-manhua-chapter-352/
 *
 * Three different places to find a series, so the rule has three arms in a fixed
 * order: a named query parameter, then the last path segment that is not a
 * chapter, then nothing at all -- host-only.
 *
 * HOST-ONLY IS A REAL ANSWER AND STRIPLINE IS WHY. `/webs/comic-next/381206` is
 * the entire reader URL: the numeric id names the CHAPTER, and two chapters of
 * one series differ in it. Keying on it would scope per chapter, which is the
 * one thing the decision rules out. So stripline falls through to the constant
 * `comic-next` segment and every stripline chapter shares one story. The known
 * cost, stated plainly rather than hidden: two different stripline titles read in
 * one session share a window. That is the OLD defect, bounded to one host
 * instead of the whole profile.
 */

/* Query parameters that name a series. Ordered, so the key does not depend on
 * the order the host happened to write the query string in.
 *
 * ONE ENTRY, because one is what the corpus evidences: the portal's `titleId`.
 * Adding a host's parameter here is a one-line change and is the right way to
 * teach this rule a new site -- but a name added on a hunch is a name that can
 * split one series into many (if it varies per chapter) or merge many into one
 * (if it is a constant), and neither failure says anything when it happens.
 * Matched case-insensitively: `titleId`, `titleid` and `TITLEID` are one site's
 * one parameter. */
const STORY_SERIES_PARAMS = ["titleid"];

/* A whole path segment that names a chapter rather than a series, so the scan
 * steps over it. Two shapes, both from the corpus:
 *
 *   digits, optionally grouped   381206 / 382913 / 163 / 0_207 / 0_207_2
 *   a chapter word plus digits   ch1-52817 / chapter-163 / vol-3
 *
 * `0_207_2` matters as much as `0_207`: paperleaf paginates one chapter across
 * four html pages, and the two must land in the same story.
 *
 * THE WORD LIST IS SHORT ON PURPOSE, and the reason is that the two ways of
 * being wrong are not symmetric. Skipping a segment that was actually the series
 * MERGES two titles into one context window -- silently, permanently, and it is
 * the exact defect this whole file exists to remove. Failing to skip one that
 * was actually a chapter splits a story at a boundary the reader can see coming.
 * So every word here has to earn its place, and an early draft that carried
 * `no` did not: `no-9` is a plausible series title, `/manga/no-9/chapter-1/` would
 * have skipped BOTH segments, and every series on that host would have shared
 * one story keyed on `manga`. `no` came from the portal's `no=163` -- which is a
 * query parameter, and never appears as a path segment at all. */
const STORY_CHAPTER_SEGMENT =
  /^(?:\d+(?:[._-]\d+)*|(?:chapter|chap|ch|episode|ep|volume|vol)[-_.]?\d+(?:[._-][a-z0-9]+)*)$/;

/* A chapter tacked onto the END of a series slug, which is what a single-segment
 * reader path is: `tiny-engine-chapter-163` and
 * `the-quiet-god-manhua-chapter-352`.
 *
 * THE DISCRIMINATOR IS A TOKEN, NOT A THRESHOLD, so there is no numeric band to
 * quote -- there is a categorical one. Across the corpus: 2 of 2 chapter tails
 * carry the literal word `chapter` before their digits, and 0 of 5 series slugs
 * contain any of these words at all (`sample-saga_abcxyz`,
 * `iron-monarch`, `tiny-engine`, `the-quiet-god-manhua`, `comic-next`). The
 * gap is the word, and a slug that merely ENDS in digits -- `robo-squad-100` is
 * the standing example -- is left alone, because a bare number after a hyphen is
 * as likely to be part of a title as to be a chapter.
 *
 * `volume` and `vol` are in the segment rule above and deliberately NOT here.
 * A whole path segment reading `vol-3` cannot be a series name; a slug ENDING in
 * `-vol-3` can be. No series is called `something-chapter-163`.
 *
 * The separator before the word is required, which is what keeps `march-3` and
 * `the-arch-9` whole: their `ch` is inside a word rather than a token of its
 * own. */
const STORY_CHAPTER_SUFFIX = /[-_](?:chapter|chap|ch|episode|ep)[-_]?\d+(?:[._-]\d+)*$/;

/* Filename extensions a reader path wears. Stripped before the segment is
 * judged, or `0_207.html` reads as a slug rather than as chapter 0_207. */
const STORY_PAGE_EXTENSION = /\.(?:html?|php|aspx?|jsp)$/;

/* How much of the readable part of a key survives into the id.
 *
 * The id reaches `story.rs::valid_id`, which caps it at 64 bytes and allows only
 * ASCII alphanumerics, `-` and `_`. The hash below carries the identity, so this
 * is purely so a log line says which story it is: 32 + 1 + 8 + at most `-r999`
 * is 46, comfortably inside the cap with room for a reset counter nobody will
 * reach. */
const STORY_MAX_SLUG = 32;

/* Everything `valid_id` refuses, folded away. A percent-encoded CJK slug becomes
 * hex-ish rubble here and that is fine -- it is stable rubble, and the hash is
 * what actually distinguishes it. */
function storySanitise(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "");
}

/* FNV-1a, 32 bits, as eight hex digits.
 *
 * Not a cryptographic hash and does not need to be: it separates at most a
 * handful of series a reader has open, and a collision costs two titles sharing
 * a context window -- exactly the pre-existing host-only fallback. Sync, so the
 * whole rule stays a pure function that a test can call without awaiting
 * crypto.subtle, which does not exist in a content script's realm anyway. */
function storyHash(text) {
  let hash = 0x811c9dc5;
  const source = String(text || "");
  for (let index = 0; index < source.length; index += 1) {
    hash ^= source.charCodeAt(index);
    // The FNV prime, multiplied as 32-bit via shifts because a plain `*` would
    // lose the low bits to a double's 53-bit mantissa.
    hash = (hash + ((hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24))) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

/* The series a path names, or "" when it names none.
 *
 * Scanned from the END rather than by position, because the depth of the reader
 * route varies -- three segments before the chapter on paperleaf, two on
 * scanhouse, none at all on readtinyengine -- while "the chapter is last" holds
 * on every one of them. */
function storySeriesFromPath(pathname) {
  const segments = String(pathname || "")
    .toLowerCase()
    .split("/")
    .map((segment) => segment.replace(STORY_PAGE_EXTENSION, ""))
    .filter(Boolean);

  for (let index = segments.length - 1; index >= 0; index -= 1) {
    const segment = segments[index];
    if (STORY_CHAPTER_SEGMENT.test(segment)) continue;
    /* Never strip a slug down to nothing. `chapter-163` as a lone segment is
     * already refused above; this covers whatever else the corpus has not
     * shown, where keeping the whole slug is strictly safer than keeping none
     * of it. */
    const trimmed = segment.replace(STORY_CHAPTER_SUFFIX, "");
    return trimmed || segment;
  }
  return "";
}

/* The story key for a page: `<host>|<series>`, and it is TOTAL -- every string
 * yields one, including one that is not a URL at all.
 *
 * The page URL, never the image URL. The two are different hosts on most
 * readers: a site like paperleaf serves its pages from `www.paperleaf.test` and
 * its images from a CDN, and `background.js` calls `hostOf()` on the IMAGE for
 * the OCR latch. A story keyed on the CDN would be one story per image host,
 * which on a site that shards across `s1`/`s2`/`s3` is a story per shard. */
function storyPageKey(pageUrl) {
  let url = null;
  try {
    url = new URL(String(pageUrl || ""));
  } catch {
    /* Not a URL at all -- an empty string, which is what `pageUrlOf` yields for
     * a message from the popup. A scheme with no host does NOT land here:
     * `about:blank` and `data:` parse fine and simply carry an empty hostname,
     * so they take the ordinary path below. Every unparseable page shares one
     * story rather than none, because the alternative is minting a fresh window
     * per page, which is the no-story arm dressed up as a story. */
    return { host: "", series: "", key: "|" };
  }
  const host = url.hostname.toLowerCase();

  let series = "";
  for (const wanted of STORY_SERIES_PARAMS) {
    for (const [name, value] of url.searchParams) {
      if (name.toLowerCase() !== wanted || !value) continue;
      series = `${wanted}=${value.toLowerCase()}`;
      break;
    }
    if (series) break;
  }
  if (!series) series = storySeriesFromPath(url.pathname);

  return { host, series, key: `${host}|${series}` };
}

/* The id sent to the server as the `story` field.
 *
 * `resets` is the popup's per-key reset counter, `{ "<key>": <n> }`. A reset
 * bumps the counter rather than minting a UUID, which is what keeps the id a
 * pure function of the URL: two lookups of the same page in the same reader
 * state must produce the same id, because that id is inside the CACHE
 * fingerprint (`cache.js::settingsFingerprint`) and a lookup that computes it
 * differently from the write is a permanent, symptomless miss. */
function storyIdFor(pageUrl, resets) {
  const { host, series, key } = storyPageKey(pageUrl);
  // The readable half: the series where there is one, the host where there is
  // not, and a constant where there is neither.
  const label = storySanitise(series || host).slice(0, STORY_MAX_SLUG).replace(/[-_]+$/, "");
  const count = Number((resets || {})[key]);
  const suffix = Number.isInteger(count) && count > 0 ? `-r${count}` : "";
  return `${label || "page"}-${storyHash(key)}${suffix}`;
}

/* Node loads this file directly for `node --test tests/story-key.test.js`. The
 * browser has no `module`, so the guard is a no-op there -- and it is a guard
 * rather than an `export` because story.js is a CLASSIC script sharing one
 * global with cache.js and background.js, exactly as seam.js is. */
if (typeof module === "object" && module !== null && module.exports) {
  module.exports = {
    STORY_SERIES_PARAMS,
    STORY_CHAPTER_SEGMENT,
    STORY_CHAPTER_SUFFIX,
    STORY_MAX_SLUG,
    storySanitise,
    storyHash,
    storySeriesFromPath,
    storyPageKey,
    storyIdFor,
  };
}
