/* Unit tests for extension/story.js -- which STORY a page belongs to.
 *
 * Run:  node --test tests/story-key.test.js
 *
 * NOT `node --test tests/` -- see the header of seam.test.js. On Node 24 a
 * bare directory argument is resolved as a MODULE and reports a failing suite
 * that never loaded this file at all.
 *
 * There is no package.json in this repo ON PURPOSE: `npx --yes web-ext ...` is
 * what lints and packages the extension, and a package.json at the root changes
 * what npx resolves. This uses node's own runner and nothing else.
 *
 * WHAT IS BEING GUARDED. The story id used to be one UUID per browser PROFILE,
 * so a single 96-pair context window spanned every title the reader ever opened
 * and carried one book's terminology into the next as established precedent. It
 * is now derived from host + series path. Two failure directions, and the tests
 * below assert BOTH because only one of them is visible:
 *
 *   fires too little  -- two chapters of one series get different keys, so the
 *                        window resets mid-volume. Visible: names drift.
 *   fires too often   -- two different series get the SAME key, so one title's
 *                        names bleed into another. INVISIBLE, and it is the old
 *                        defect wearing a new hat.
 *
 * The site names and slugs below are invented. Each URL exercises one shape of
 * reader URL -- path depth, chapter token, query parameter -- which is all the
 * rule reads. Where a case needs a SECOND series on one host, it reuses a slug
 * from another shape and is labelled `CONSTRUCTED` at the site.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  STORY_SERIES_PARAMS,
  STORY_MAX_SLUG,
  storySanitise,
  storyHash,
  storySeriesFromPath,
  storyPageKey,
  storyIdFor,
} = require("../extension/story.js");

const ROOT = path.join(__dirname, "..");
const BACKGROUND_JS = fs.readFileSync(
  path.join(ROOT, "extension", "background.js"),
  "utf8"
);
const STORY_JS = fs.readFileSync(path.join(ROOT, "extension", "story.js"), "utf8");
const MANIFEST = JSON.parse(
  fs.readFileSync(path.join(ROOT, "extension", "manifest.json"), "utf8")
);

/* The whole corpus, and the whole basis of the rule: six reader shapes. */
const PAPERLEAF_1 =
  "https://www.paperleaf.test/comic/chapter/sample-saga_abcxyz/0_901.html";
// One chapter is FOUR html pages; page 2 of chapter 0_901.
const PAPERLEAF_1B =
  "https://www.paperleaf.test/comic/chapter/sample-saga_abcxyz/0_901_2.html";
// Stripline 381206 and 382913 are chapters 1 and 2 of ONE series; the chapter
// id is the whole reader URL.
const STRIPLINE_CH1 = "https://www.stripline.test/webs/comic-next/381206";
const STRIPLINE_CH2 = "https://www.stripline.test/webs/comic-next/382913";
const PORTAL = "https://comic.portal.test/webtoon/detail?titleId=700001&no=163&week=thu";
const PORTAL_NO_WEEK = "https://comic.portal.test/webtoon/detail?titleId=700001&no=163";
const SCANHOUSE = "https://scanhouse.test/read/iron-monarch/ch1-52817/";
const TINYENGINE = "https://w27.readtinyengine.test/tiny-engine-chapter-163/";
const QUIETGOD = "https://quietgod.test/the-quiet-god-manhua-chapter-352/";

const key = (url) => storyPageKey(url).key;

/* ----------------------------------------------------- what the rule extracts */

test("every corpus shape resolves to the series the reader is actually in", () => {
  assert.deepEqual(storyPageKey(PAPERLEAF_1), {
    host: "www.paperleaf.test",
    series: "sample-saga_abcxyz",
    key: "www.paperleaf.test|sample-saga_abcxyz",
  });

  assert.equal(storyPageKey(SCANHOUSE).series, "iron-monarch");
  assert.equal(storyPageKey(TINYENGINE).series, "tiny-engine");
  assert.equal(storyPageKey(QUIETGOD).series, "the-quiet-god-manhua");

  // The series is a QUERY parameter here and `no` is the episode. A path-only
  // rule would read `detail` and give every portal title one story.
  assert.equal(storyPageKey(PORTAL).series, "titleid=700001");

  /* Stripline names NO series anywhere in the reader URL -- `381206` is the
   * chapter. This is the host-only fallback doing its job: the segment it lands
   * on is constant across the whole site, which is what the decision asks for
   * and is strictly better than keying on the chapter. */
  assert.equal(storyPageKey(STRIPLINE_CH1).series, "comic-next");
});

/* --------------------------------------------- direction 1: it must not SPLIT */

test("two chapters of one series share a key", () => {
  // Chapters 1 and 2 of one series on stripline.
  assert.equal(key(STRIPLINE_CH1), key(STRIPLINE_CH2));

  // The two html pages of paperleaf chapter 0_901. A chapter is paginated
  // across four of them and all four are the same reading.
  assert.equal(key(PAPERLEAF_1), key(PAPERLEAF_1B));

  // CONSTRUCTED from the paperleaf shape: a later chapter of the same comic.
  const paperleafLater =
    "https://www.paperleaf.test/comic/chapter/sample-saga_abcxyz/0_902.html";
  assert.equal(key(PAPERLEAF_1), key(paperleafLater));

  // Portal episode 163, with and without the `week` parameter the site adds
  // when you arrive from the weekday list. Same episode, same series.
  assert.equal(key(PORTAL), key(PORTAL_NO_WEEK));
  // CONSTRUCTED: the next episode of the same portal title, and its list page.
  assert.equal(key(PORTAL), key("https://comic.portal.test/webtoon/detail?titleId=700001&no=164"));
  assert.equal(key(PORTAL), key("https://comic.portal.test/webtoon/list?titleId=700001"));

  // CONSTRUCTED from the scanhouse and readtinyengine shapes.
  assert.equal(key(SCANHOUSE), key("https://scanhouse.test/read/iron-monarch/ch2-52818/"));
  assert.equal(
    key(TINYENGINE),
    key("https://w27.readtinyengine.test/tiny-engine-chapter-164/")
  );
});

/* -------------------------------------- direction 2: it must not MERGE titles
 *
 * This is the half that has no symptom, so it is the half that has to be
 * asserted. The corpus held one series per host, so the second series in each
 * pair is CONSTRUCTED -- a slug from somewhere else in the corpus, dropped into
 * the shape of the host under test.
 */

test("two different series on the same host do not share a key", () => {
  // paperleaf: its own comic, against the `iron-monarch` slug in its shape.
  assert.notEqual(
    key(PAPERLEAF_1),
    key("https://www.paperleaf.test/comic/chapter/iron-monarch/0_901.html")
  );

  // scanhouse, whose series sits mid-path.
  assert.notEqual(key(SCANHOUSE), key("https://scanhouse.test/read/tiny-engine/ch1-1/"));

  // Both slugs are corpus slugs; only their sharing a host is constructed.
  assert.notEqual(
    key(TINYENGINE),
    key("https://w27.readtinyengine.test/the-quiet-god-manhua-chapter-352/")
  );

  // portal, where the discriminator is the query parameter.
  assert.notEqual(key(PORTAL), key("https://comic.portal.test/webtoon/detail?titleId=700002&no=1"));

  // The same slug on two different hosts is two stories: the host is in the key
  // and the same slug on another site is another translation, by someone else.
  assert.notEqual(key(SCANHOUSE), key("https://quietgod.test/read/iron-monarch/ch1-1/"));
});

/* ------------------------------------------------- the chapter-strip's silence
 *
 * The suffix rule is the one that reaches inside a slug, so it is the one that
 * can quietly eat part of a title. The discriminator is a TOKEN, not a
 * threshold, so there is no numeric band to quote -- there is a categorical one,
 * and this is it: across the corpus 2 of 2 chapter tails carry the literal word
 * `chapter` before their digits (`-chapter-163`, `-chapter-352`) and 0 of 5
 * series slugs contain any chapter word at all.
 */

test("the chapter strip stays silent on every corpus series slug", () => {
  const untouched = [
    // The five corpus slugs.
    "sample-saga_abcxyz",
    "iron-monarch",
    "tiny-engine",
    "the-quiet-god-manhua",
    "comic-next",
    /* And the shapes that a looser rule eats. A bare number after a hyphen is
     * as likely to be part of a title as a chapter -- `robo-squad-100` is the
     * standing example -- and `part`/`vol` are chapter units as whole segments
     * but ordinary title words as suffixes. */
    "robo-squad-100",
    "strange-journey-part-4",
    "42-forty-two",
    // The separator before the word is required, or these lose their tails.
    "march-3",
    "the-arch-9",
  ];
  for (const slug of untouched) {
    assert.equal(
      storySeriesFromPath(`/manga/${slug}/chapter-12/`),
      slug,
      `${slug} is a title, not a chapter`
    );
  }
});

test("the chapter strip fires on the two corpus chapter tails", () => {
  assert.equal(storySeriesFromPath("/tiny-engine-chapter-163/"), "tiny-engine");
  assert.equal(storySeriesFromPath("/the-quiet-god-manhua-chapter-352/"), "the-quiet-god-manhua");
});

test("a segment that is nothing but a chapter never becomes the series", () => {
  /* Each of these is skipped and the scan walks on. If any were taken as a
   * series, every chapter of the title would be its own story -- which is the
   * per-chapter scoping the decision rules out. */
  for (const chapter of ["381206", "0_901", "0_901_2", "163", "ch1-52817", "chapter-163", "vol-3"]) {
    assert.equal(
      storySeriesFromPath(`/read/iron-monarch/${chapter}/`),
      "iron-monarch",
      `${chapter} is a chapter`
    );
  }

  /* ...and the counter-population, which an earlier draft of the segment rule
   * got wrong. `no` was in the chapter-word list because the portal's episode
   * parameter is `no=163` -- but `no` never appears as a path SEGMENT, and
   * `no-9` is a plausible series title. With `no` in the list both segments here are
   * skipped and the key collapses onto `manga`, merging every series on the
   * host into one context window. */
  assert.equal(storySeriesFromPath("/manga/no-9/chapter-1/"), "no-9");
  assert.equal(storySeriesFromPath("/manga/part-time-hero/chapter-1/"), "part-time-hero");
});

/* ---------------------------------------------------------------- totality */

test("every input yields a key, including things that are not URLs", () => {
  for (const junk of ["", "not a url", "about:blank", "javascript:0", null, undefined, 42]) {
    const resolved = storyPageKey(junk);
    assert.equal(typeof resolved.key, "string");
    assert.ok(resolved.key.length > 0, `${String(junk)} produced no key`);
    // ...and an id, which is what actually reaches the server.
    assert.ok(storyIdFor(junk, {}).length > 0);
  }

  // A page with no path at all -- the site's front page -- is host-only rather
  // than keyless.
  assert.equal(key("https://www.stripline.test/"), "www.stripline.test|");
});

/* ---------------------------------------------------- what the server accepts
 *
 * `server/src/story.rs::valid_id` refuses an id that is empty, over 64 bytes, or
 * anything but ASCII alphanumerics, `-` and `_`. An id it refuses is a request
 * translated with no context at all, silently, so the shape is checked here
 * rather than discovered on a page.
 */

const MAX_ID_LEN = 64;
const validId = (id) => id.length > 0 && id.length <= MAX_ID_LEN && /^[A-Za-z0-9_-]+$/.test(id);

test("every id the rule can mint is one the server will accept", () => {
  const urls = [
    PAPERLEAF_1, STRIPLINE_CH1, PORTAL, SCANHOUSE, TINYENGINE, QUIETGOD,
    "https://www.stripline.test/",
    // A percent-encoded CJK slug, which is what a Chinese host serves when the
    // title is not romanised. CONSTRUCTED.
    "https://example.test/manga/%E4%BA%91%E6%B5%B7/chapter-1/",
    // Absurd but reachable: a very long slug, and a query value nobody bounded.
    `https://example.test/manga/${"a".repeat(400)}/chapter-1/`,
    `https://example.test/webtoon/detail?titleId=${"9".repeat(400)}`,
    "not a url",
  ];
  for (const url of urls) {
    const id = storyIdFor(url, {});
    assert.ok(validId(id), `${url} minted ${JSON.stringify(id)}`);
  }
  // ...and with a reset counter on top, which is the longest form.
  const longest = `https://example.test/manga/${"a".repeat(400)}/chapter-1/`;
  const bumped = storyIdFor(longest, { [storyPageKey(longest).key]: 999 });
  assert.ok(validId(bumped), JSON.stringify(bumped));
  assert.ok(bumped.endsWith("-r999"));
});

test("the readable half is trimmed but the hash still separates", () => {
  /* Truncation is what makes the hash load-bearing: two slugs sharing their
   * first 32 characters would otherwise be one story. */
  const a = `https://example.test/manga/${"a".repeat(STORY_MAX_SLUG)}-one/chapter-1/`;
  const b = `https://example.test/manga/${"a".repeat(STORY_MAX_SLUG)}-two/chapter-1/`;
  assert.notEqual(storyPageKey(a).key, storyPageKey(b).key);
  assert.notEqual(storyIdFor(a, {}), storyIdFor(b, {}));

  // The hash is over the WHOLE key, so the host is in it even when the readable
  // half is the series alone.
  assert.notEqual(
    storyIdFor("https://one.test/read/iron-monarch/ch1-1/", {}),
    storyIdFor("https://two.test/read/iron-monarch/ch1-1/", {})
  );
  assert.equal(storyHash("a"), storyHash("a"));
  assert.notEqual(storyHash("a"), storyHash("b"));
  assert.equal(storySanitise("Title Id=700001"), "title-id-700001");
});

/* ------------------------------------------------------------------- resets */

test("a reset moves one series and leaves the others alone", () => {
  const stripline = storyPageKey(STRIPLINE_CH1).key;
  const before = storyIdFor(STRIPLINE_CH1, {});
  const portalBefore = storyIdFor(PORTAL, {});

  const resets = { [stripline]: 1 };
  assert.notEqual(storyIdFor(STRIPLINE_CH1, resets), before);
  // The other chapter of the same series moves with it -- it is one story.
  assert.equal(storyIdFor(STRIPLINE_CH2, resets), storyIdFor(STRIPLINE_CH1, resets));
  // ...and the title the reader also has open does not.
  assert.equal(storyIdFor(PORTAL, resets), portalBefore);

  // A second press moves it again, and a missing or junk counter is no reset.
  assert.notEqual(storyIdFor(STRIPLINE_CH1, { [stripline]: 2 }), storyIdFor(STRIPLINE_CH1, resets));
  for (const junk of [undefined, null, 0, -1, NaN, {}, "x"]) {
    assert.equal(storyIdFor(STRIPLINE_CH1, { [stripline]: junk }), before, String(junk));
  }
  assert.equal(storyIdFor(STRIPLINE_CH1, undefined), before);

  /* A counter that came back from storage as a STRING still counts, and the
   * lenient direction is the deliberate one: a value that round-tripped as
   * `"1"` means one reset, and reading it as none would resurrect the very
   * story the reader pressed the button to forget. */
  assert.equal(storyIdFor(STRIPLINE_CH1, { [stripline]: "1" }), storyIdFor(STRIPLINE_CH1, resets));
});

/* -------------------------------------------------- who calls it, and with what
 *
 * The same static shape cache-key.test.js guards for the host, one field along.
 * `cfg.storyId` is inside `settingsFingerprint`, and it is DERIVED rather than
 * stored -- so a caller that resolves `config()` without a page URL keys every
 * page under the empty story while `translate` keys under the real one. The
 * `urls` index is queried with `IDBKeyRange.only`: no near miss, no fallback, a
 * total and symptomless miss on every page for ever.
 */

// The full argument text of a `config(...)` call, paren-balanced, because the
// arguments themselves contain calls -- `config(hostOf(url), pageUrl)`.
function configArgs(line) {
  const at = line.search(/\bconfig\s*\(/);
  if (at < 0) return null;
  const open = line.indexOf("(", at);
  let depth = 0;
  for (let index = open; index < line.length; index += 1) {
    if (line[index] === "(") depth += 1;
    else if (line[index] === ")") {
      depth -= 1;
      if (depth === 0) return line.slice(open + 1, index);
    }
  }
  return null;
}

// The same text, split on its TOP-LEVEL commas, so `config(hostOf(url), pageUrl)`
// is two arguments and `config(hostOf(a, b))` is one.
function splitArgs(args) {
  if (!args || !args.trim()) return [];
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let index = 0; index < args.length; index += 1) {
    const character = args[index];
    if (character === "(" || character === "[" || character === "{") depth += 1;
    else if (character === ")" || character === "]" || character === "}") depth -= 1;
    else if (character === "," && depth === 0) {
      parts.push(args.slice(start, index));
      start = index + 1;
    }
  }
  parts.push(args.slice(start));
  return parts.map((part) => part.trim());
}

// One whole function, from its signature to the `}` in column 0 that closes it.
// Every function in background.js is written at top level in that style.
function functionSource(source, signature) {
  const at = source.indexOf(signature);
  assert.ok(at >= 0, `${signature} is gone from background.js`);
  const end = source.indexOf("\n}", at);
  assert.ok(end > at, `could not find the end of ${signature}`);
  return source.slice(at, end);
}

const IDENTIFIER = /[A-Za-z_$][A-Za-z0-9_$]*/g;

test("every settingsFingerprint caller in background.js resolves a page URL too", () => {
  const lines = BACKGROUND_JS.split(/\r?\n/);
  const offenders = [];

  lines.forEach((line, index) => {
    if (!/settingsFingerprint\s*\(/.test(line)) return;
    if (/^\s*(\/\/|\*|\/\*)/.test(line)) return;

    let args = null;
    for (let back = index; back >= Math.max(0, index - 12); back -= 1) {
      args = configArgs(lines[back]);
      if (args !== null) break;
    }
    // Two arguments: the image host for the OCR latch, the page URL for the
    // story. Depth-aware, so `config(hostOf(url), pageUrl)` counts as two and
    // `config(hostOf(a, b))` would not.
    let depth = 0;
    let count = args && args.trim() ? 1 : 0;
    for (const character of args || "") {
      if (character === "(") depth += 1;
      else if (character === ")") depth -= 1;
      else if (character === "," && depth === 0) count += 1;
    }
    if (count < 2) offenders.push(`${index + 1}: ${line.trim()}  [config(${args})]`);
  });

  assert.deepEqual(
    offenders,
    [],
    "these call settingsFingerprint on a config() resolved without a page URL, " +
      "so they key under the empty story and can never match what translate() " +
      "stored:\n  " + offenders.join("\n  ")
  );
});

/* ...and COUNTING the arguments is not enough, which is the hole the test above
 * left open. `config(hostOf(url), url)` in `cacheLookup` has two arguments and
 * passes it -- while `url` is the IMAGE's url, whose host is a CDN on every
 * reader in the corpus, never the page's host. Lookup would key under one
 * story and `translate` under another, for ever, on every page: `cacheGetByUrl`
 * queries the `urls` index with `IDBKeyRange.only`, so there is no near miss and
 * no fallback and nothing to see. The second argument has to be the SAME text at
 * every site, and it has to be a name the image url did not produce. */

test("the page URL handed to config() is one name, and never the image url's", () => {
  const lines = BACKGROUND_JS.split(/\r?\n/);
  const calls = [];

  lines.forEach((line, index) => {
    if (/^\s*(\/\/|\*|\/\*)/.test(line)) return;
    // The declaration `async function config(host, pageUrl)` is not a call site.
    if (/\bfunction\s+config\b/.test(line)) return;
    const parts = splitArgs(configArgs(line));
    if (parts.length < 2) return;
    calls.push({ at: index + 1, first: parts[0], second: parts[1], text: line.trim() });
  });

  /* cacheLookup, translate and seamJoin. Not an upper bound -- a fourth caller
   * is welcome and is checked like the rest -- but if the scan finds fewer than
   * three then either a call site lost its page URL or this parser stopped
   * working, and both must be loud rather than vacuously green. */
  assert.ok(
    calls.length >= 3,
    `expected at least the three story-bearing config() call sites, found ${calls.length}:\n  ` +
      calls.map((call) => `${call.at}: ${call.text}`).join("\n  ")
  );

  const names = [...new Set(calls.map((call) => call.second))];
  assert.deepEqual(
    names,
    [names[0]],
    "the second argument to config() differs between call sites, so they resolve " +
      "the story from different URLs and can never agree on a cache key:\n  " +
      calls.map((call) => `${call.at}: config(${call.first}, ${call.second})`).join("\n  ")
  );

  for (const call of calls) {
    assert.match(
      call.second,
      /^[A-Za-z_$][A-Za-z0-9_$]*$/,
      `${call.at}: the page URL must be a plain name, not the expression ${JSON.stringify(call.second)}`
    );
    /* And it must not be a name the FIRST argument built its host out of. That
     * first argument is `hostOf(<the image url>)`, so any identifier shared with
     * it -- `url`, `top.url` -- means the story is being keyed on the CDN. */
    const fromImage = new Set(call.first.match(IDENTIFIER) || []);
    assert.ok(
      !fromImage.has(call.second),
      `${call.at}: config(${call.first}, ${call.second}) resolves the story from the ` +
        `IMAGE url. The page and the image are different hosts on every real ` +
        `reader, so this keys one story per CDN shard and misses the writer's key ` +
        `on every page, silently.`
    );
  }
});

/* --------------------------------------------------------- the reset's wiring
 *
 * `browser.tabs.query` is the ONE external browser API this change introduces,
 * and the rule it feeds is pure and fully tested while the wiring around it was
 * not tested at all. Replacing `(tab && tab.url) || ""` with `""` leaves every
 * assertion above green: the reader presses Reset, the popup answers "Reset. The
 * next page of this series starts fresh.", and the counter that moves belongs to
 * the empty story `"|"` while the series they are actually reading keeps all 96
 * pairs. Nothing anywhere says so.
 */

test("resetStory resets the series in the tab the popup is open over", () => {
  const body = functionSource(BACKGROUND_JS, "async function resetStory(");

  assert.match(
    body,
    /browser\.tabs\s*\.query\s*\(/,
    "resetStory no longer asks which tab it is over, so it cannot know which series to reset"
  );
  const query = /browser\.tabs\s*\.query\s*\(([\s\S]*?)\)/.exec(body)[1];
  assert.match(query, /active\s*:\s*true/, `tabs.query(${query}) is not asking for the active tab`);
  assert.match(
    query,
    /currentWindow\s*:\s*true/,
    `tabs.query(${query}) is not scoped to the window the popup is in, so a background window's tab can win`
  );
  assert.match(
    body,
    /\.catch\s*\(/,
    "a tabs.query the extension may not answer must not reject the whole reset"
  );

  // THE assertion. The page URL is the tab's, and is not conjured from nothing.
  const binding = /\bconst\s+pageUrl\s*=\s*([^;]+);/.exec(body);
  assert.ok(binding, "resetStory no longer binds a pageUrl");
  assert.match(
    binding[1],
    /\btab\b[\s\S]*\.url\b/,
    `resetStory derives its page URL from \`${binding[1].trim()}\`, which is not the ` +
      `tab's URL. It will clear and bump some other series than the one on screen, ` +
      `and still report success.`
  );

  /* ...and that one URL is what BOTH halves of the reset use: the id told to the
   * server, and the key whose counter is bumped. Two URLs here would forget one
   * story on the server and start a different one in the browser. */
  assert.match(body, /storyPageKey\s*\(\s*pageUrl\s*\)/, "the bumped key is not the tab's");
  assert.match(body, /storyIdFor\s*\(\s*pageUrl\s*,/, "the id sent to the server is not the tab's");
  assert.match(
    body,
    /storage\.local\.set\s*\(\s*\{\s*storyResets/,
    "the bumped counter is never persisted, so the next page recomputes the old id"
  );
});

/* ------------------------------------------------ the manifest is a dependency
 *
 * story.js is a CLASSIC script sharing one global with background.js, and that
 * link exists ONLY in manifest.json. Delete the entry and `node --test` stays
 * green -- this file `require`s story.js directly -- and `web-ext lint` stays
 * 0/0/0, because it does not resolve identifiers across the scripts a background
 * page shares a global with. What actually happens is that `config()` calls
 * `storyIdFor` unconditionally, no other file defines it, so config() throws
 * ReferenceError and EVERY message dies: not just translation, but the first
 * `pageEnabled` a content script sends on every page load.
 */

test("the manifest loads story.js into the background page, ahead of background.js", () => {
  const used = ["storyIdFor", "storyPageKey"].filter((name) =>
    new RegExp(`\\b${name}\\s*\\(`).test(BACKGROUND_JS)
  );
  assert.ok(used.length > 0, "background.js no longer calls story.js at all");

  for (const name of used) {
    assert.ok(
      new RegExp(`function\\s+${name}\\b`).test(STORY_JS),
      `background.js calls ${name}, which story.js does not define`
    );
    /* If background.js defined it too, the two classic scripts would collide on
     * one name in one global -- so story.js really is the only definer, and the
     * manifest entry really is the only thing that supplies it. */
    assert.ok(
      !new RegExp(`function\\s+${name}\\b`).test(BACKGROUND_JS),
      `${name} is declared in background.js as well as story.js`
    );
  }

  const scripts = (MANIFEST.background && MANIFEST.background.scripts) || [];
  assert.ok(
    scripts.includes("story.js"),
    `manifest background.scripts is ${JSON.stringify(scripts)} -- story.js is absent, so ` +
      `config() throws ReferenceError: ${used[0]} is not defined, on the very first ` +
      `message the background page handles.`
  );
  assert.ok(
    scripts.indexOf("story.js") < scripts.indexOf("background.js"),
    `manifest background.scripts is ${JSON.stringify(scripts)} -- story.js must load ` +
      `before background.js, whose top-level lexical bindings share story.js's global`
  );
});

test("the story id is derived, never stored", () => {
  /* A stored id is a second source of truth that can disagree with the derived
   * one, and the disagreement is the cache miss above. The only `storyId` that
   * may appear in a storage write is the REMOVAL of the legacy per-profile
   * one. */
  const writes = BACKGROUND_JS.match(/storage\.local\.set\([^)]*storyId/g) || [];
  assert.deepEqual(writes, [], "background.js still persists a story id");
  assert.ok(
    /storage\.local\.remove\(\[[^\]]*"storyId"/.test(BACKGROUND_JS),
    "the legacy per-profile storyId is never cleaned up"
  );
  // The UUID mint is what made it per-profile. It should be gone entirely.
  assert.ok(!/randomUUID/.test(BACKGROUND_JS), "a story id is still being minted");
});

test("the parameter list is the documented one", () => {
  // A guard against a name being added on a hunch: every entry has to be
  // lowercase, because the lookup lowercases what it compares against, and a
  // mixed-case entry would silently never match.
  for (const name of STORY_SERIES_PARAMS) {
    assert.equal(name, name.toLowerCase(), `${name} can never match`);
  }
  assert.deepEqual(STORY_SERIES_PARAMS, ["titleid"]);
});
