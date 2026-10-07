/* BireLate - the translation cache.
 *
 * Loaded into the background script and into the popup. Both are the same
 * moz-extension origin, so both see the same database; the popup only reads,
 * apart from its Clear button.
 *
 * IndexedDB rather than storage.local: a translated page is 1-3 MB, and
 * storage.local would keep it as a base64 data URL -- a third larger, JSON
 * parsed in full on every read, and read in full by every content script at
 * document_idle. IndexedDB stores a Blob natively, and it fires no
 * storage.onChanged, which the content scripts sweep the page on.
 */

const CACHE_DB = "birelate-cache";
const CACHE_DB_VERSION = 1;
const CACHE_PAGES = "pages";
const CACHE_META = "meta";

const CACHE_MAX_AGE_MS = 24 * 60 * 60 * 1000;

// Defaults for the two LRU caps, shared so the popup and the background script
// cannot disagree about what an unset setting means.
const CACHE_DEFAULT_ENTRIES = 500;
const CACHE_DEFAULT_BYTES = 1024 ** 3;

// One popup panel's worth. The store holds far more than anyone reads.
const CACHE_LIST_LIMIT = 30;

/* A data: URL is the translated image itself -- megabytes of key for a lookup
 * that can only ever miss. A blob: URL dies with the document that made it.
 * Neither is worth storing, and neither is anything absurdly long. */
const CACHE_MAX_URL = 2048;

// A CDN that rotates its paths would otherwise grow one entry's list forever.
const CACHE_MAX_URLS = 8;

/* A promise, never a connection. The background script is an event page and is
 * suspended after a short idle, which takes this whole global with it, so the
 * reopen is usually free. The handlers below cover the cases where the
 * connection dies while the page is still alive. */
let cacheDb = null;

function cacheOpen() {
  if (cacheDb) return cacheDb;

  cacheDb = new Promise((resolve, reject) => {
    const req = indexedDB.open(CACHE_DB, CACHE_DB_VERSION);

    req.onupgradeneeded = () => {
      const db = req.result;

      if (!db.objectStoreNames.contains(CACHE_PAGES)) {
        const pages = db.createObjectStore(CACHE_PAGES, { keyPath: "key" });

        // Expiry is a range delete over this rather than a scan of the store.
        pages.createIndex("at", "at");

        /* [used, bytes] and not plain "used": a key cursor over a compound
         * index hands back the size inside the key, so the eviction pass totals
         * the store without reading one record -- and therefore without
         * touching one blob. */
        pages.createIndex("lru", ["used", "bytes"]);

        /* multiEntry indexes each member of the array separately, so a URL
         * resolves straight to its entry. The members carry the settings
         * fingerprint because the same URL translated into two languages is two
         * entries, and tier 1 must not hand back the wrong one. */
        pages.createIndex("urls", "urls", { multiEntry: true });
      }

      if (!db.objectStoreNames.contains(CACHE_META)) {
        db.createObjectStore(CACHE_META, { keyPath: "name" });
      }
    };

    /* Another context is still holding an older version open. The
     * onversionchange handler below makes this unreachable in practice; without
     * the warning it would present as a translate that never returns. */
    req.onblocked = () => console.warn("[koharu] cache upgrade blocked");

    req.onsuccess = () => {
      const db = req.result;
      // Step aside for an upgrade instead of blocking it, and reopen on demand.
      db.onversionchange = () => {
        db.close();
        cacheDb = null;
      };
      // Site data cleared underneath us, or the profile closed the connection.
      db.onclose = () => {
        cacheDb = null;
      };
      resolve(db);
    };

    /* Dropping the promise matters: a cached rejected one would turn a single
     * transient failure into a cache that stays dead for the life of the page. */
    req.onerror = () => {
      cacheDb = null;
      reject(req.error);
    };
  });

  return cacheDb;
}

/* Awaiting an IndexedDB request from inside its own transaction is safe: the
 * continuation runs in the microtask checkpoint of the success event, while the
 * transaction is still active. Awaiting anything else -- a digest, a fetch, a
 * FileReader -- lets the transaction commit and the next call throws. Every
 * function below builds its whole value first and opens its transaction last. */
const cacheRequest = (req) =>
  new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });

const cacheTxDone = (tx) =>
  new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onabort = () => reject(tx.error || new Error("cache transaction aborted"));
    tx.onerror = () => reject(tx.error || new Error("cache transaction failed"));
  });

/* ------------------------------------------------------------------- keying */

/* Exactly the fields the request carries. `llm` is folded in as "" when blank
 * because translate() omits the form field entirely in that case -- a blank and
 * an absent model are the same request and must be the same key. JSON rather
 * than a separator so no value can be confused for a field boundary. */
function settingsFingerprint(cfg) {
  return JSON.stringify([
    cfg.targetLanguage,
    cfg.ocr,
    /* Collapsed while the artwork is being kept, because then no inpainter runs
     * and all four choices render byte-identical pixels. Discriminating on a
     * setting that cannot change the output costs a re-fetch, a re-hash and a
     * fresh GPU run per page -- and stores a second copy of the same image
     * against the entry caps. Same rule as `llm || ""` below: requests that
     * cannot differ must not key differently. */
    cfg.keepArt ? "" : cfg.inpainting,
    cfg.provider,
    cfg.llm || "",
    /* Changes which regions are lettered: the server's implausible-text gate
     * runs its two script rules only when a language is declared, so the same
     * page under "zh" and under "" are two different renders. Coerced, because
     * a store written before this existed must key the same as an explicit
     * empty declaration.
     *
     * `cfg.profile` is deliberately ABSENT even though the request carries it:
     * the server ships the field inert, so two requests differing
     * only in profile render byte-identical pixels, and requests that cannot
     * differ must not key differently -- this list's own rule, same as the
     * keepArt collapse above. The patch that gives the server a behavioral
     * consumer of `profile` MUST add it here in the same change, or a flipped
     * profile would serve the other profile's render for 24 hours. */
    cfg.sourceLanguage || "",
    /* Every setting that changes the returned pixels has to be in here, and this
     * one changes them more visibly than any of the others: with it on the
     * artwork under the bubbles survives, with it off it is painted out. Left
     * out, both tiers would key the two renders identically and flipping the
     * switch would appear to do nothing for 24 hours -- in whichever direction
     * happened to be cached first. Coerced, because `undefined` from a store
     * written by an older version must key the same as an explicit false. */
    Boolean(cfg.keepArt),
    /* Segment context changes the PROMPT, so it changes the words and therefore
     * the pixels -- the same reason `sourceLanguage` and `storyId` are here. The
     * measured difference is not subtle: on the same page and the same binary,
     * one ability name came out as a transliteration with it on and as a
     * literal English phrase with it off. Left out of the key, flipping it
     * would appear to do nothing for 24 hours, in whichever direction happened
     * to be cached first.
     *
     * THE READER-FACING TOGGLE HAS BEEN REMOVED and nothing in the extension
     * sets this any more, so it is a constant `false`. It stays
     * in the key deliberately: the server still accepts `segment_context` and
     * still has `--segment-context`, and re-adding a control without re-adding
     * this line is exactly how the 24-hour bug above comes back. Deleting it
     * would also re-key every cached page for one cycle to save one boolean.
     * Coerced for `keepArt`'s reason: a store written before this setting
     * existed must key the same as an explicit false. */
    Boolean(cfg.segmentContext),
    /* A story changes the prompt, so it changes the pixels: the same page read
     * inside a story and on its own are two different translations and must not
     * share a key.
     *
     * THIS FIELD IS WHY THE STORY ID HAS TO BE A PURE FUNCTION OF THE PAGE URL.
     * It is derived, not stored (`story.js::storyIdFor`), and both the tier-1
     * lookup and the write compute it independently -- so if the two were handed
     * different page URLs, or one were handed none, they would key under
     * different stories for ever. `cacheGetByUrl` queries the `urls` index with
     * `IDBKeyRange.only`, so there is no near miss and no fallback: it would be
     * a total, silent miss on every page. tests/story-key.test.js pins the call
     * sites for exactly that reason.
     *
     * The one-off cost of moving from a per-profile UUID to a per-series id is
     * real and worth naming: every entry stored under the old id becomes
     * unreachable and ages out on the ordinary 24 h boundary. One re-translation
     * per page the reader revisits inside that window, once, ever.
     *
     * Note what this deliberately does NOT capture. Within one story the
     * carried context grows page by page, so page five's prompt differs from
     * what it would have been at page two -- and a hit here serves the render
     * made when that page was first translated. That is the intended reading:
     * re-viewing a page you already translated in this story should show you
     * the translation you were given, not a new one. */
    typeof cfg.storyId === "string" ? cfg.storyId : "",
  ].concat(
    /* The per-series glossary changes the prompt, so it changes the pixels --
     * the story id's own rule, one mechanism over. APPENDED ONLY WHEN
     * NON-EMPTY, and that asymmetry is load-bearing: a glossary-less reader
     * must key byte-identically to every entry written before the field
     * existed ("a store written before this existed must key the same" -- the
     * rule every coercion above already follows), so absence cannot be a ninth
     * "" element, it has to be no element at all. Minted in `config()` beside
     * `cfg.storyId` (glossary.js::glossaryFingerprint over the session-stored
     * terms) so the tier-1 lookup and the write read the same value -- the
     * same discipline, for the same silent-permanent-miss failure. */
    typeof cfg.glossaryFingerprint === "string" && cfg.glossaryFingerprint
      ? [cfg.glossaryFingerprint]
      : []
  ));
}

const cacheKey = (srcHash, fingerprint) => `${srcHash}:${fingerprint}`;

function cacheUrlKey(url, fingerprint) {
  if (typeof url !== "string" || !url || url.length > CACHE_MAX_URL) return null;
  if (url.startsWith("data:") || url.startsWith("blob:")) return null;
  return `${url}\n${fingerprint}`;
}

/* Hashed here and never in the content script: crypto.subtle exists only in a
 * secure context, and a content script runs in the page's realm -- undefined on
 * every http:// manga host. */
async function sha256Hex(data) {
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/* The hourly sweep is housekeeping, not the freshness guarantee: the browser can
 * be shut for three days, and the startup sweep races the first lookup. So every
 * read decides for itself, on the same boundary the sweep deletes on. */
const cacheFresh = (entry, now) =>
  Boolean(entry) && entry.at >= now - CACHE_MAX_AGE_MS;

/* The page's <img> is handed a data: URL, exactly as a fresh translation is. An
 * object URL minted here would carry this extension's origin and a web page
 * cannot load that; one minted in the content script would need a Blob to
 * survive runtime.sendMessage, which is the very thing the bytes already travel
 * as an array to avoid. Serving one shape both ways also means a hit renders
 * exactly as well as a fresh translation, with no second path to get wrong --
 * and nothing to revoke. */
function cacheDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("could not read the translated image"));
    reader.readAsDataURL(blob);
  });
}

/* ------------------------------------------------------------------ lookups */

/* Tier 1. The whole point of the cache is a page the user is rereading, and the
 * point of this tier is that such a page does not refetch its images at all. */
async function cacheGetByUrl(url, fingerprint) {
  const member = cacheUrlKey(url, fingerprint);
  if (!member) return null;

  const db = await cacheOpen();
  const tx = db.transaction(CACHE_PAGES, "readonly");
  const req = tx
    .objectStore(CACHE_PAGES)
    .index("urls")
    .openCursor(IDBKeyRange.only(member));
  const now = Date.now();

  return await new Promise((resolve, reject) => {
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const cursor = req.result;
      if (!cursor) {
        resolve(null);
        return;
      }
      /* A cursor rather than a get(), because two records can carry the same
       * URL: one that aged out keeps its URL until the sweep collects it, while
       * the retranslation that replaced it is filed under a fresh hash. Without
       * this walk, tier 1 would miss for as long as the stale one survived. */
      if (cacheFresh(cursor.value, now)) {
        resolve(cursor.value);
        return;
      }
      cursor.continue();
    };
  });
}

// Tier 2. Correct when the host rotates its URLs on the same picture.
async function cacheGetByKey(key) {
  const db = await cacheOpen();
  const tx = db.transaction(CACHE_PAGES, "readonly");
  const entry = await cacheRequest(tx.objectStore(CACHE_PAGES).get(key));
  return cacheFresh(entry, Date.now()) ? entry : null;
}

/* ------------------------------------------------------------------ writing */

/* `missed` and `cut` are the server's miss report, kept with the entry so a hit
 * warns exactly as the live translation did. Without them the warning would
 * survive one view of a page and disappear on the next, which is a worse signal
 * than none: the reader learns that a partial page sometimes says so.
 *
 * `edges` is seam.js's summary of which boxes this page's translation found
 * against its top and bottom edges -- two short lists of four numbers and the
 * detector's class name, and deliberately not the region list, because `bytes`
 * below is `blob.size` alone and anything stored beside the blob is invisible to
 * the LRU byte cap. An entry written before the class was summarised simply has
 * boxes without one, and seam.js reads that as "we do not know" rather than as a
 * refusal -- see `seamIsSfx`.
 *
 * Those lists can also hold an EDGE HINT: a box the server found at a page edge
 * and refused as a region for scoring under its text floor, reported on its own
 * channel and folded into this summary by `seamEdges` because the seam is its
 * only consumer. It is indistinguishable from any other box once it is here, and
 * that is deliberate -- it is four numbers, it is read by exactly the same
 * geometry, and nothing outside seam.js ever sees this field. Nothing extra is
 * stored and nothing extra has to be remembered on the read side. It is
 * here rather than in memory so that a REVISITED webtoon can still be joined:
 * the second view of a strip is two tier-1 hits that never reach the server, and
 * without this the neighbour test would have nothing to test.
 *
 * `seamed` is which of this page's two edges have already had a rejoined bubble
 * painted into the blob. The correction is baked into the image rather than
 * cached separately -- a seam spans two pages and belongs to neither, so it has
 * no URL to be filed under and two fresh entries sharing one URL member are
 * resolved by index order, not recency. Baking it in means a revisit is an
 * ordinary hit that is already correct, and this field is what stops the join
 * being paid for a second time. */
async function cachePut({ srcHash, fingerprint, url, blob, source, ms, missed, dropped, cut,
                          duplicateIds, outOfRangeIds, cutFound, stillCut,
                          edges, terms, boxes, edits }) {
  const member = cacheUrlKey(url, fingerprint);
  const now = Date.now();
  const entry = {
    key: cacheKey(srcHash, fingerprint),
    srcHash,
    urls: member ? [member] : [],
    settings: fingerprint,
    at: now,
    // Distinct from `at`, which the 24h expiry owns: refreshing `at` on a hit
    // would keep a popular page alive forever.
    used: now,
    ms,
    /* Both blobs, so the LRU byte cap sees what the entry actually weighs --
     * the `edges` comment above already establishes that anything stored
     * beside `blob` is otherwise invisible to it, and `source` is the one
     * field heavy enough for that to matter. */
    bytes: blob.size + (source ? source.size : 0),
    hits: 0,
    missed: missed || null,
    /* On the entry for exactly the reason `missed` is, and more sharply: a drop
     * leaves nothing on the page pointing at its own cause, so a warning that
     * appeared on a page's first view and vanished on its second would leave the
     * reader with an unexplained region and no way to find out why. A string
     * like "4/13", so an old entry with no field reads as nothing to say. */
    dropped: dropped || null,
    cut: Boolean(cut),
    /* The server's repeated-id and out-of-range-id counters, kept for exactly
     * the reason `missed` and `cut` are:
     * they name the CAUSE of a miss, and a cause that appeared on a page's first
     * view and vanished on its second would be a worse signal than none. Two
     * small integers, so unlike `edges` there is nothing here to weigh against
     * `bytes` being `blob.size` alone.
     *
     * Not in the fingerprint, and the rule is the same one `joinSlices` is the
     * exception to: the fingerprint names everything that changes the PIXELS of
     * a request, and these are outputs of the run, not inputs to it. Two pages
     * differing only in how many ids the model repeated are the same request. */
    duplicateIds: Number(duplicateIds) || 0,
    outOfRangeIds: Number(outOfRangeIds) || 0,
    /* The cut verdict, same reasoning as the pair above: outputs of the
     * run, never fingerprint inputs, and a still-cut bubble whose warning
     * vanished on the second view would read as the page having healed. */
    cutFound: Number(cutFound) || 0,
    stillCut: Number(stillCut) || 0,
    edges: edges || null,
    /* The pinnable terms: the term-shaped regions' boxes and strings, so
     * a cache hit can still answer an alt-click. Small by construction -- most
     * pages carry none -- and absent on every pre-feature entry, which reads
     * as nothing to pin. */
    terms: Array.isArray(terms) && terms.length ? terms : null,
    /* The box editor's substrate and its cumulative edits: the render's region
     * rectangles, and the add/remove lists this render was made under, so a
     * revisit can open the editor and a retry cannot resurrect a deleted box.
     * Rects only -- four numbers a region -- so nothing here weighs against
     * `bytes` the way `source` does. */
    boxes: Array.isArray(boxes) && boxes.length ? boxes : null,
    edits: edits || null,
    seamed: [],
    blob,
    /* The bytes the translation was made FROM -- `srcHash`'s own preimage,
     * stored verbatim so a later retry can re-roll after the host's url has
     * died (signed urls expire; blob: urls are revoked; the page then shows
     * only OUR lettering). Absent on every pre-feature entry, which a retry
     * reads as "translate it fresh first". */
    source: source || null,
  };

  const db = await cacheOpen();
  const tx = db.transaction(CACHE_PAGES, "readwrite");
  tx.objectStore(CACHE_PAGES).put(entry);
  await cacheTxDone(tx);
}

/* Replace a stored page with the same page after a seam has been painted into
 * it, and record which edge that was.
 *
 * A whole `cachePut` would be wrong here and the reason is not obvious: it
 * rebuilds the record from scratch, so `urls` collapses to the one member it is
 * handed and `hits` goes back to zero -- destroying every alternate URL the
 * picture has accumulated, which is exactly what makes tier 1 cheap on a host
 * that rotates its paths. This is a mutation, so it reads, edits and puts.
 *
 * `at` is deliberately untouched: the seam is a correction to a translation
 * already made, not a new one, and refreshing it would let a page that is joined
 * on every revisit outlive the 24-hour window forever. `used` is not touched
 * either -- the hit that will follow this is what moves it. */
/* The seamed-edge list after a join that repaired `side`, which is ONE edge or
 * SEVERAL.
 *
 * Several because a slice a run passes THROUGH was cut at its top and at its
 * bottom, and one join repaired both. Marking only one of them would have the
 * next visit re-run the whole run to fix an edge already fixed -- and marking
 * them in two calls would be two transactions over the same record, where the
 * second's read can precede the first's write.
 *
 * Pure, named and separate from the transaction so it can be tested: the
 * multi-edge branch is the only thing standing between a run and an unbounded
 * re-join on every revisit, and it is invisible to every other test. */
function cacheSeamedWith(seamed, side) {
  const marks = Array.isArray(side) ? side : [side];
  const next = Array.isArray(seamed) ? [...seamed] : [];
  for (const mark of marks) if (!next.includes(mark)) next.push(mark);
  return next;
}

async function cacheMarkSeamed(key, blob, side) {
  const db = await cacheOpen();
  const tx = db.transaction(CACHE_PAGES, "readwrite");
  const store = tx.objectStore(CACHE_PAGES);
  const entry = await cacheRequest(store.get(key));

  if (entry) {
    if (blob) {
      entry.blob = blob;
      entry.bytes = blob.size;
    }
    entry.seamed = cacheSeamedWith(entry.seamed, side);
    store.put(entry);
  }

  await cacheTxDone(tx);
  return Boolean(entry);
}

/* What a hit does to its entry: move it to the back of the eviction queue, and
 * record the URL that found it. A URL that resolved through the hash tier is
 * worth writing down -- the next reload then resolves through tier 1, which is
 * the tier that skips the refetch. */
async function cacheTouch(key, url, fingerprint) {
  const member = cacheUrlKey(url, fingerprint);

  const db = await cacheOpen();
  const tx = db.transaction(CACHE_PAGES, "readwrite");
  const store = tx.objectStore(CACHE_PAGES);
  const entry = await cacheRequest(store.get(key));

  if (entry) {
    entry.used = Date.now();
    entry.hits = (entry.hits || 0) + 1;
    if (member && !entry.urls.includes(member)) {
      entry.urls = [...entry.urls, member].slice(-CACHE_MAX_URLS);
    }
    store.put(entry);
  }

  await cacheTxDone(tx);
}

/* Two counters, read by the popup. They live here rather than in storage.local
 * because every content script listens to storage.onChanged unfiltered and
 * sweeps its page on any change -- a counter bumped on every translate would
 * rescan every image in every open tab. */
async function cacheBump(field) {
  const db = await cacheOpen();
  const tx = db.transaction(CACHE_META, "readwrite");
  const store = tx.objectStore(CACHE_META);
  const stats = (await cacheRequest(store.get("stats"))) || {
    name: "stats",
    hits: 0,
    misses: 0,
  };
  stats[field] = (stats[field] || 0) + 1;
  store.put(stats);
  await cacheTxDone(tx);
}

/* ----------------------------------------------------------------- sweeping */

/* IDBIndex has no delete(), and objectStore.delete(range) ranges over the
 * primary key only, so a delete by timestamp is a cursor plus a delete by
 * primaryKey. openKeyCursor and never openCursor: the values carry the blobs and
 * an expiry pass must not read a single one of them. */
function cacheDropExpired(store, cutoff) {
  return new Promise((resolve, reject) => {
    let dropped = 0;
    const req = store
      .index("at")
      .openKeyCursor(IDBKeyRange.upperBound(cutoff, true));
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const cursor = req.result;
      if (!cursor) {
        resolve(dropped);
        return;
      }
      store.delete(cursor.primaryKey);
      dropped += 1;
      cursor.continue();
    };
  });
}

/* The caps the 24h window alone does not give: a heavy session can read a
 * thousand pages in one sitting. The cursor is ordered least-recently-used
 * first, which is the order to evict in, and its key is [used, bytes] so the
 * running total costs no record reads. */
function cacheTrim(store, maxEntries, maxBytes) {
  return new Promise((resolve, reject) => {
    const rows = [];
    const req = store.index("lru").openKeyCursor();
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor) {
        rows.push({ primaryKey: cursor.primaryKey, bytes: cursor.key[1] });
        cursor.continue();
        return;
      }

      let count = rows.length;
      let bytes = rows.reduce((sum, row) => sum + row.bytes, 0);
      let dropped = 0;
      for (const row of rows) {
        if (count <= maxEntries && bytes <= maxBytes) break;
        store.delete(row.primaryKey);
        count -= 1;
        bytes -= row.bytes;
        dropped += 1;
      }
      resolve(dropped);
    };
  });
}

async function cacheSweep({ maxEntries, maxBytes }) {
  const db = await cacheOpen();
  const tx = db.transaction(CACHE_PAGES, "readwrite");
  const store = tx.objectStore(CACHE_PAGES);
  const expired = await cacheDropExpired(store, Date.now() - CACHE_MAX_AGE_MS);
  const evicted = await cacheTrim(store, maxEntries, maxBytes);
  await cacheTxDone(tx);
  return { expired, evicted };
}

// The caps alone, cheap enough to run after every stored translation.
async function cacheEnforceCaps({ maxEntries, maxBytes }) {
  const db = await cacheOpen();
  const tx = db.transaction(CACHE_PAGES, "readwrite");
  const evicted = await cacheTrim(
    tx.objectStore(CACHE_PAGES),
    maxEntries,
    maxBytes
  );
  await cacheTxDone(tx);
  return evicted;
}

/* ------------------------------------------------------------------- readout */

function cacheTotals(store) {
  return new Promise((resolve, reject) => {
    let count = 0;
    let bytes = 0;
    const req = store.index("lru").openKeyCursor();
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const cursor = req.result;
      if (!cursor) {
        resolve({ count, bytes });
        return;
      }
      count += 1;
      bytes += cursor.key[1];
      cursor.continue();
    };
  });
}

async function cacheSummary() {
  const db = await cacheOpen();
  const tx = db.transaction([CACHE_PAGES, CACHE_META], "readonly");
  const totals = await cacheTotals(tx.objectStore(CACHE_PAGES));
  const stats = await cacheRequest(tx.objectStore(CACHE_META).get("stats"));
  return {
    count: totals.count,
    bytes: totals.bytes,
    hits: (stats && stats.hits) || 0,
    misses: (stats && stats.misses) || 0,
  };
}

// Newest first, and without the blobs: holding a record would keep its image
// alive for as long as the popup shows the list.
async function cacheList(limit = CACHE_LIST_LIMIT) {
  const db = await cacheOpen();
  const tx = db.transaction(CACHE_PAGES, "readonly");
  const req = tx.objectStore(CACHE_PAGES).index("at").openCursor(null, "prev");

  return await new Promise((resolve, reject) => {
    const rows = [];
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const cursor = req.result;
      if (!cursor) {
        resolve(rows);
        return;
      }
      const entry = cursor.value;
      rows.push({
        at: entry.at,
        ms: entry.ms,
        bytes: entry.bytes,
        hits: entry.hits || 0,
      });
      if (rows.length >= limit) {
        resolve(rows);
        return;
      }
      cursor.continue();
    };
  });
}

/* The counters go with the entries. They count hits against pages that are
 * being deleted, so keeping them would describe a cache that no longer exists. */
async function cacheClear() {
  const db = await cacheOpen();
  const tx = db.transaction([CACHE_PAGES, CACHE_META], "readwrite");
  tx.objectStore(CACHE_PAGES).clear();
  tx.objectStore(CACHE_META).clear();
  await cacheTxDone(tx);
}
