/* BireLate - background.
 *
 * All traffic to the local BireLate server happens here. The extension holds
 * host permission for 127.0.0.1, so this fetch is not subject to CORS, and the
 * server never has to relax its own origin policy.
 */

const DEFAULTS = {
  enabledEverywhere: false,
  enabledHosts: [],
  serverUrl: "http://127.0.0.1:8765",
  targetLanguage: "en-US",
  /* THE BILINGUAL ENGINE IS THE DEFAULT, AND `manga-ocr` IS EARNED.
   *
   * A `manga-ocr` default would mean every new host reads its first 40 CJK
   * characters with a Japanese-only engine -- including Chinese and Korean
   * ones, where it fabricates. The engine is chosen only on positive
   * evidence of Japanese AND paged (`script-latch.js`), and until that evidence
   * exists the bilingual engine reads everything.
   *
   * The direction follows the asymmetry `SCRIPT_KANA_SHARE` documents: paddle
   * (the bilingual engine when this was measured) wrongly costs ~2 s a page,
   * is loud and reversible, and measured zero accuracy cost; manga-ocr wrongly
   * costs silent fabrication over artwork. */
  /* HunyuanOCR is the bilingual default; PaddleOCR-VL stays selectable as the
   * reserve. */
  ocr: "hunyuan-ocr-1.5",
  /* Let the script decide the OCR engine, per host. See `scriptOcr`.
   *
   * On by default because the failure it prevents is SILENT: `manga-ocr` cannot
   * represent Chinese -- its 6,144-entry character vocabulary has none of the
   * commonest simplified characters and no byte fallback -- but it does not fail
   * or return empty. It emits fluent Japanese, measured, including real Japanese
   * words over Chinese artwork. `confidence` is `None`, `untranslated` is empty,
   * and the LLM turns the fabrication into confident English. A reader who does
   * not know to change a dropdown has no way to find out.
   *
   * Turned off the moment the reader picks an engine by hand: an explicit choice
   * must never be second-guessed by a heuristic. */
  ocrAuto: true,
  /* Per-host latch, `{ "host": "hunyuan-ocr-1.5" }`, written by `noteScript`.
   * Host rather than story: the story is a reading session and is minted fresh
   * whenever the reader asks, while the language of a site does not change. */
  ocrByHost: {},
  /* The other half of that latch, `{ "host": "zh" }`, from the same evidence.
   * Sent as `source_language` so the server's implausible-text gate can run its
   * two script rules; without it the server runs only the four rules that need
   * no language. Empty string means the evidence was too thin to say. */
  languageByHost: {},
  inpainting: "lama",
  /* Keep the artwork: run detection, OCR and translation but not inpainting, so
   * the English is drawn over the untouched page instead of over an erased
   * patch. The mask the inpainters are handed is a 288x288 segmentation head
   * upsampled to page size plus an unfeathered ~10px dilation, so on free-
   * standing text over art it routinely eats line work that was never damaged.
   * This is the escape hatch for those pages; `inpainting` still chooses which
   * model runs when it is off. */
  keepArt: false,
  /* Join a balloon whose SHAPE is cut by the slice boundary while its text
   * sits wholly on one side (the TOUCHES class), by probing the neighbour
   * slice's edge band for the balloon's white continuation. ON by default:
   * two must-not-fire control boundaries stayed un-planned with the flag
   * armed, verified on the wire and in the rendered pages. No popup checkbox
   * yet: a storage override is the off arm. */
  touchJoin: true,
  /* Matches the server's own default. These two disagreeing is not a cosmetic
   * mismatch: the server pins its backend for the life of the process and
   * answers a request naming a different one with a 409, so a fresh install
   * used to fail its very first translate. "ollama" here was a leftover from
   * when Ollama was the plan; this project settled on local Gemma-4. */
  provider: "local",
  llm: "",
  token: "",
  /* The reader's explicit layout pick per PAGE host, `{ "host": "manga" }` or
   * `"webtoon"`. Absent means Auto: the latch's format evidence decides, and
   * before it has any the profile is simply undeclared.
   *
   * PAGE host, not image host, on purpose: one CDN serves many sites, so an
   * explicit pick keyed on the image host would apply one site's answer to
   * another. The latch maps below stay image-host keyed -- evidence about who
   * served the pixels belongs to who served them -- and `resolveSite`
   * (script-latch.js) owns the precedence between the two.
   *
   * This FOLDED the old `joinSlices` global checkbox: the seam now runs
   * unless the resolved profile is `manga`. Auto preserves the old default --
   * the seam was always additionally self-scoped by its exact width test, so
   * ordinary manga pages never joined anyway. A `joinSlices` key left in an
   * old store is dead. */
  profileByHost: {},
  /* The reader's explicit language pick per PAGE host, `{ "host": "ko" }`.
   * Beats `languageByHost`'s latched evidence; same page-host keying and for
   * the same reason as `profileByHost`. */
  languageOverrideByHost: {},
  /* The latch's format verdict per IMAGE host, `{ "host": "strip" }` or
   * `"paged"`, written by `noteScript` in the same storage write as the engine
   * and language so the three cannot disagree. It is what lets the popup show
   * "Auto (detected: webtoon)" and lets `resolveSite` derive a profile before
   * the reader has picked one. */
  formatByHost: {},
  /* Which IMAGE host served the last translated picture on each PAGE host,
   * `{ "pageHost": "imageHost" }`. Display plumbing only: the popup knows the
   * page host and the latch is keyed on the image host, and on a CDN-served
   * reader the two differ. Written by `translate()` when it changes. */
  imageHostForPage: {},
  cacheMaxEntries: CACHE_DEFAULT_ENTRIES,
  cacheMaxBytes: CACHE_DEFAULT_BYTES,
  /* How many times the reader has pressed "Reset story context" on each series,
   * `{ "<host>|<series>": <n> }`. See `resetStory`.
   *
   * THERE IS NO `storyId` HERE ANY MORE, and its absence is the whole patch.
   * It used to be a UUID minted once per browser profile and stored for ever,
   * so one 96-pair context window spanned every title the reader ever opened.
   * The id is now COMPUTED from the page's URL by `story.js::storyIdFor`, which
   * makes it a pure function of where the reader is rather than a fact about
   * their profile -- so no default is needed and a stored one would be a second,
   * silently disagreeing source of truth. A legacy value left in storage.local
   * is simply overwritten in `config()`; `onInstalled` removes it for tidiness,
   * not for correctness. */
  storyResets: {},
};

const GIB = 1024 ** 3;

/* Asked at most once per event-page lifetime, and only when there is no token. */
let tokenFromHost = null;

/* Kana and Han, the whole of the language test.
 *
 * Japanese prose is written with kana between its Han; Chinese has none. That
 * one property separates the two scripts more cleanly than anything else
 * available to a browser, and it needs no model, no extra request and no GPU --
 * it is a codepoint count over a string the reply already carries.
 *
 * Measured on test material, dialogue regions only: a Japanese volume is
 * 71.6% kana with 203 of 213 pages carrying some; a Chinese chapter is 2.0%
 * with only 13 of 245 pages carrying any, all of them 1-5 character effects.
 * A 36x separation with nothing in between. */
const KANA = /[぀-ゟ゠-ヿ]/gu;
const HAN = /[一-鿿]/gu;
/* Counted only to name the language, never to choose the engine: PaddleOCR-VL
 * reads Hangul and Han alike, so this cannot move `SCRIPT_KANA_SHARE`'s decision
 * and is deliberately kept out of its denominator. */
const HANGUL = /[가-힣ᄀ-ᇿ]/gu;

/* `SCRIPT_MIN_CHARS`, `SCRIPT_KANA_SHARE`, `BILINGUAL_OCR` and `latchDecision`
 * moved to `script-latch.js`, which `manifest.json` loads before this file.
 * They went together because they are one decision, and a test can drive it
 * there with no `browser.*` -- this file cannot be required at all, since it
 * reads globals from `cache.js`. */

/* `hostOf` is declared further down beside the host-permission helpers, which is
 * where it belongs. It is only ever called at request time, long after the
 * module has finished evaluating, so the temporal dead zone never applies.
 * Reusing it rather than adding a second one also keeps the latch keyed on
 * `hostname` -- no port -- which is the same key `enabledHosts` uses. */

/* MIGRATION: the bilingual default changed from `paddleocr-vl-1.6` to
 * `hunyuan-ocr-1.5`, and the per-host latch persists the OLD name for every
 * host visited before the flip -- so without this, the flip silently never
 * happens on any host the reader already uses.
 *
 * Rewriting every paddle entry is safe because `ocrByHost` is written by
 * `noteScript` ALONE -- it is always the heuristic's "not Japanese, use the
 * bilingual engine" verdict, never a reader's explicit pick. An explicit pick
 * lives in `cfg.ocr` with `ocrAuto` off, and this touches neither. The latched
 * VERDICT (bilingual vs manga-ocr) survives; only the engine filling the
 * bilingual role changes. Runs once per browser start; a no-op after the first.
 */
async function migrateLatchedBilingual() {
  try {
    const stored = await browser.storage.local.get("ocrByHost");
    const latched = stored.ocrByHost || {};
    const moved = {};
    for (const [host, engine] of Object.entries(latched)) {
      if (engine === "paddleocr-vl-1.6") moved[host] = "hunyuan-ocr-1.5";
    }
    if (Object.keys(moved).length) {
      await browser.storage.local.set({ ocrByHost: { ...latched, ...moved } });
      console.debug(
        `[birelate] migrated ${Object.keys(moved).length} latched host(s) to hunyuan-ocr-1.5`
      );
    }
  } catch (e) {
    console.warn("[birelate] latch migration failed", e);
  }
}
migrateLatchedBilingual();

/* MIGRATION: `formatByHost` did not exist when the latch fired on
 * older stores, so a latched host has an engine and a language but no format --
 * and the profile selector would show "Auto (detected: --)" on a site the latch
 * long since decided. One direction is recoverable: `manga-ocr` is only
 * reachable through positive `paged` evidence (`latchDecision`), so a host
 * latched to it was a paged host. Bilingual hosts stay unknown -- the engine is
 * compatible with every format, so nothing about the format can be read back
 * out of it, and inventing `strip` here would hand `resolveSite` a webtoon
 * verdict the evidence never gave. Runs once per browser start; a no-op after
 * the first. */
async function migrateLatchedFormats() {
  try {
    const stored = await browser.storage.local.get(["ocrByHost", "formatByHost"]);
    const latched = stored.ocrByHost || {};
    const formats = stored.formatByHost || {};
    const moved = {};
    for (const [host, engine] of Object.entries(latched)) {
      if (engine === "manga-ocr" && !formats[host]) moved[host] = "paged";
    }
    if (Object.keys(moved).length) {
      await browser.storage.local.set({ formatByHost: { ...formats, ...moved } });
      console.debug(
        `[birelate] derived paged format for ${Object.keys(moved).length} manga-ocr host(s)`
      );
    }
  } catch (e) {
    console.warn("[birelate] format migration failed", e);
  }
}
migrateLatchedFormats();

/* Accumulates script evidence for one host and latches an engine once there is
 * enough of it.
 *
 * Counted over DIALOGUE regions only. `onomatopoeia` is excluded because every
 * stray-kana region in the measured Chinese chapter was a short sound effect,
 * and effects are exactly where a Japanese-trained reader's hallucinations land.
 * The detector's own worst input must not be its evidence.
 *
 * The latch is written once and then left alone. It is not re-evaluated per page
 * because the answer cannot change -- a site does not switch language mid-
 * chapter -- and because each change of `cfg.ocr` is a `PipelineConfig` field,
 * so flipping it costs a `Pipeline::reload`: an emptied `Residency`, every model
 * evicted to be re-profiled, a cold page at best and a 507 on a contended card
 * at worst. Once per host, never per page. */
async function noteScript(host, regions, shape) {
  if (!host || !Array.isArray(regions) || !regions.length) return;
  const cfg = { ...DEFAULTS, ...(await browser.storage.local.get(null)) };
  if (!cfg.ocrAuto) return;
  const latched = cfg.ocrByHost || {};
  if (latched[host]) {
    /* THE LATCH KEEPS WATCHING. A bare `return` here would let one wrong
     * early verdict poison every later series on the host, forever -- observed
     * when a Japanese+paged latch put manga-ocr under a Chinese chapter and
     * its Japanese styling looked like a seam defect.
     *
     * A latched host now accumulates a DISSENT probe, the same shape and the
     * same readiness floor as the first decision. Agreement at readiness
     * CLEARS the probe -- that reset is the hysteresis: a mixed or noisy
     * stream keeps starting over and never flips, so the one reload a
     * re-latch costs is paid only at a genuine series boundary, where every
     * page pushes the same way. The tally mirrors the unlatched path below
     * rather than sharing a helper so each path's reasoning stays beside its
     * own numbers. */
    const dissent = cfg.ocrDissent || {};
    const seen = dissent[host] || { kana: 0, han: 0, hangul: 0 };
    for (const region of regions) {
      if (region && region.label === "onomatopoeia") continue;
      const source = (region && region.source) || "";
      seen.kana += (source.match(KANA) || []).length;
      seen.han += (source.match(HAN) || []).length;
      seen.hangul = (seen.hangul || 0) + (source.match(HANGUL) || []).length;
    }
    if (shape === "strip") seen.strip = (seen.strip || 0) + 1;
    else if (shape === "paged") seen.paged = (seen.paged || 0) + 1;
    /* NOT the first latch's "strip wins if ever seen": that rule makes paged
     * evidence EARN its engine, which is right for a first verdict and wrong
     * for overturning one -- a single strip sighting would condemn the whole
     * dissent window and flip a paged host's engine on one page's noise. A
     * re-decision must earn the CHANGE, so the window's format is its own
     * majority, and a tie keeps the standing verdict -- which is what lets a
     * genuinely mixed host keep agreeing, resetting, and never flipping. */
    const latchedFormat = (cfg.formatByHost || {})[host] || "";
    const format =
      (seen.strip || 0) > (seen.paged || 0)
        ? "strip"
        : (seen.paged || 0) > (seen.strip || 0)
          ? "paged"
          : latchedFormat;
    const decided = latchDecision(seen, format);
    if (!decided.ready) {
      await browser.storage.local.set({ ocrDissent: { ...dissent, [host]: seen } });
      return;
    }
    const next = { ...dissent };
    delete next[host];
    const agrees =
      decided.engine === latched[host] &&
      decided.language === ((cfg.languageByHost || {})[host] || "") &&
      (decided.format || "") === ((cfg.formatByHost || {})[host] || "");
    if (agrees) {
      await browser.storage.local.set({ ocrDissent: next });
      return;
    }
    await browser.storage.local.set({
      ocrByHost: { ...latched, [host]: decided.engine },
      languageByHost: { ...(cfg.languageByHost || {}), [host]: decided.language },
      formatByHost: { ...(cfg.formatByHost || {}), [host]: decided.format },
      ocrDissent: next,
    });
    console.debug(
      `[birelate] script re-latch ${host}: ${latched[host]}/` +
        `${(cfg.languageByHost || {})[host] || "undeclared"} -> ${decided.engine}/` +
        `${decided.language || "undeclared"} on ${decided.total} chars of dissent`
    );
    return;
  }

  const probe = cfg.ocrProbe || {};
  const seen = probe[host] || { kana: 0, han: 0, hangul: 0 };
  for (const region of regions) {
    if (region && region.label === "onomatopoeia") continue;
    const source = (region && region.source) || "";
    seen.kana += (source.match(KANA) || []).length;
    seen.han += (source.match(HAN) || []).length;
    // Absent from a probe written by an older version, hence the `|| 0`.
    seen.hangul = (seen.hangul || 0) + (source.match(HANGUL) || []).length;
  }
  /* The page's own shape, tallied alongside the script evidence and resolved in
   * the same write. Same lifetime, same key, same "once per host, never per
   * page" guarantee the engine already has -- which is what keeps a mixed host
   * from flipping `cfg.ocr` mid-chapter and paying a `Pipeline::reload` for it.
   * `|| 0` for the same reason `hangul` has one: a probe written by an older
   * version carries neither field. */
  if (shape === "strip") seen.strip = (seen.strip || 0) + 1;
  else if (shape === "paged") seen.paged = (seen.paged || 0) + 1;

  /* STRIP WINS A TIE, and wins outright if it was ever seen.
   *
   * `manga-ocr` is what `paged` unlocks, so `paged` is the answer that must be
   * EARNED. A host that ever looked like a strip is one where the classifier's
   * cheap error already lands correctly, and a mixed host resolves to the engine
   * that reads both scripts. Only a host that has looked like discrete pages
   * every single time it was observed can reach `manga-ocr`. */
  const format = (seen.strip || 0) > 0 ? "strip" : (seen.paged || 0) > 0 ? "paged" : "";

  const decided = latchDecision(seen, format);
  if (!decided.ready) {
    await browser.storage.local.set({ ocrProbe: { ...probe, [host]: seen } });
    return;
  }

  /* ONE decision, one site. `latchDecision` returns the engine AND the language
   * together, and both are stored from here unchanged.
   *
   * This block must not recompute the language a second time with a
   * different test -- `hangul > seen.han` rather than `latchDecision`'s
   * `hangul > kana + han`. The two genuinely disagree: on
   * `{kana: 3, han: 20, hangul: 22}` the function returns `zh` and such a
   * block would store `ko`. **The stored one wins**, so `script-latch.js`'s
   * tests would be pinning a value the caller threw away.
   *
   * Why the language must NOT be re-derived from the engine anywhere else: the
   * mapping runs `engine === "manga-ocr" ? "ja" : ...`, so anything that forces
   * the engine away from manga-ocr silently re-declares a Japanese page as `zh`
   * -- and `labels.rs`'s script rule then refuses its kana as wrong-script. On
   * a measured Japanese webtoon that is 559 of 583 regions. Change the engine
   * only AFTER this, never inside it. */
  const { engine, language, share } = decided;
  const hangul = seen.hangul || 0;
  const next = { ...probe };
  delete next[host];
  await browser.storage.local.set({
    ocrByHost: { ...latched, [host]: engine },
    languageByHost: { ...(cfg.languageByHost || {}), [host]: language },
    /* The format verdict, kept rather than discarded now that the profile axis
     * reads it (`resolveSite`). Same write as the engine and language so the
     * three latched facts cannot disagree; `decided.format` can be "" when the
     * evidence latched on script alone, and "" resolves to an undeclared
     * profile, not a guessed one. */
    formatByHost: { ...(cfg.formatByHost || {}), [host]: decided.format },
    ocrProbe: next,
  });
  console.debug(
    `[birelate] script latch ${host}: kana ${seen.kana}/${decided.total} = ${share.toFixed(3)}` +
      ` -> ${engine}, hangul ${hangul} vs han ${seen.han} -> ${language || "undeclared"}`
  );
}

/* `host` is the IMAGE's host, which is what the OCR latch is keyed on; `pageUrl`
 * is the URL of the page the reader is looking at, which is what the STORY is
 * keyed on. They are different things on most readers -- a site like paperleaf
 * serves its pages from `www.paperleaf.test` and its images from a CDN -- so
 * neither can be derived from the other and both have to be handed in. */
async function config(host, pageUrl) {
  const cfg = { ...DEFAULTS, ...(await browser.storage.local.get(null)) };
  /* Applied HERE, before `settingsFingerprint(cfg)` sees the object, for the
   * same reason the story id is resolved here: `cfg.ocr` is part of the
   * fingerprint, so choosing the engine any later would compute the cache key
   * under one engine and send the request under another. Two different renders
   * would key identically -- the exact bug the fingerprint exists to prevent. */
  /* The reader's explicit picks are PAGE-host keyed (the host they can see and
   * the popup can name); the latch stays IMAGE-host keyed. `resolveSite` owns
   * the precedence and the engine re-derivation -- it is the composed predicate,
   * extracted so a test can drive the join rather than the halves.
   *
   * `cfg.sourceLanguage` is resolved on the same line of reasoning and in the
   * same place as `cfg.ocr`, because it is in the fingerprint for the same
   * reason: it changes which regions get lettered, so choosing it any later
   * would compute the cache key under one language and send the request under
   * another. `cfg.profile` is deliberately NOT in the fingerprint while the
   * server ships it inert -- requests that cannot differ must not key
   * differently (`settingsFingerprint`'s own rule); the patch that gives the
   * server a behavioral consumer of `profile` must add it to the fingerprint
   * in the same change. */
  const pageHost = hostOf(pageUrl);
  const site = resolveSite(
    {
      profile: (cfg.profileByHost || {})[pageHost] || "",
      language: (cfg.languageOverrideByHost || {})[pageHost] || "",
    },
    {
      engine: (host && (cfg.ocrByHost || {})[host]) || "",
      language: (host && (cfg.languageByHost || {})[host]) || "",
      format: (host && (cfg.formatByHost || {})[host]) || "",
    },
    Boolean(cfg.ocrAuto)
  );
  if (site.ocr) cfg.ocr = site.ocr;
  cfg.sourceLanguage = site.language;
  cfg.profile = site.profile;
  /* Story context is ON by default, and this is where the story is CHOSEN.
   *
   * It is resolved HERE rather than inside `translate` for a reason that is not
   * obvious and would be a silent cache bug: `settingsFingerprint(cfg)` runs
   * against this same object before the request is built, and the story id is
   * part of the fingerprint. Resolving later would compute the key under "" and
   * send the request under a real id, so a page would be stored against a key
   * no later read can produce -- a permanent miss that looks like the cache
   * simply not working.
   *
   * Every reader gets a story without asking for one because that is what the
   * measurement supports: over 40 pages of one volume in reading order, the
   * carried pairs held one romanisation for every recurring name, while the
   * same pages translated alone spelled the same surname three ways and once
   * dropped it for a literal translation of its kanji.
   *
   * AND THAT MEASUREMENT IS WHY THE ID IS PER SERIES. It was taken over 40
   * pages of ONE VOLUME, so it is evidence for a per-work window -- but the id
   * was a UUID minted once per browser profile, so the window spanned every
   * title the reader opened until they pressed Reset, carrying one book's
   * terminology into the next as established precedent. `storyIdFor` derives it
   * from host + series path instead, falling back to host-only where the reader
   * URL names no series (stripline's `/webs/comic-next/381206` names only the
   * chapter). Not per chapter: a chapter boundary resets the window at exactly
   * the point a romanisation has finally settled.
   *
   * A pure function of the URL, which is what lets the lookup and the write
   * agree without either storing anything. */
  cfg.storyId = storyIdFor(pageUrl, cfg.storyResets);
  /* The per-series glossary, minted HERE beside the story id and for the same
   * reason: `glossaryFingerprint` joins `settingsFingerprint`'s key, so
   * resolving it any later would compute the key under "" and send the
   * request under real terms -- the lookup and the write must read the same
   * value from the same place. Keyed on the RESET-INVARIANT series key rather
   * than the story id: "New story" abandons the carried pairs, never the
   * reader's terms -- the server's glossary store is a deliberate sibling of
   * the story store on exactly that reasoning (`server/src/glossary.rs`) --
   * so a reset simply re-installs the same terms under the fresh id on the
   * next translate. `storage.session` on purpose, twice over: it dies when
   * the browser closes (by design, mirroring the server store dying with the
   * server process), and it survives event-page suspension,
   * which a module-level variable would not. */
  cfg.glossarySeriesKey = storyPageKey(pageUrl).key;
  const glossaryKey = glossaryStorageKey(cfg.glossarySeriesKey);
  const glossaryStored = await browser.storage.session.get(glossaryKey).catch(() => ({}));
  cfg.glossaryTerms = Array.isArray(glossaryStored[glossaryKey])
    ? glossaryStored[glossaryKey]
    : [];
  cfg.glossaryFingerprint = glossaryFingerprint(cfg.glossaryTerms);
  if (cfg.token) return cfg;

  /* A fresh profile has no token, and neither does one whose storage was
   * cleared -- storage.local goes with the profile. The token itself survives
   * in the native host's state, outside the browser profile, so ask the native
   * host for it instead of making the reader paste it again.
   *
   * Silent when there is no host: that is the ordinary state for anyone who
   * runs serve.ps1 by hand, and they paste the token themselves. */
  tokenFromHost ??= browser.runtime
    .sendNativeMessage(NATIVE_HOST, { command: "token" })
    .then(async (reply) => {
      /* Not cached: before the first server start there is no token yet, and
       * a cached "" would 401 status and warmup until the background unloads. */
      if (!reply || !reply.ok || !reply.token) {
        tokenFromHost = null;
        return "";
      }
      await browser.storage.local.set({ token: reply.token });
      return reply.token;
    })
    .catch(() => "");

  const token = await tokenFromHost;
  return token ? { ...cfg, token } : cfg;
}

const base = (cfg) => cfg.serverUrl.replace(/\/$/, "");

/* /health is deliberately the only unauthenticated route. Everything else
 * carries the shared secret, because any web page can also reach 127.0.0.1 and
 * the loopback interface alone is not an authorisation. */
const authHeaders = (cfg) => (cfg.token ? { "X-Koharu-Token": cfg.token } : {});

const gib = (bytes) => `${(bytes / GIB).toFixed(1)} GB`;

const fail = (err) => ({
  ok: false,
  error: String((err && err.message) || err),
  kind: (err && err.kind) || "server",
});

async function detail(res) {
  // Server errors are plain text and the meaning lives in the first 120 chars.
  return (await res.text().catch(() => "")).slice(0, 120);
}

/* --------------------------------------------------------------- status cache
 *
 * The popup shows a live countdown to the server's idle unload. Polling once a
 * second for that would be absurd, so the popup is handed a deadline and ticks
 * against its own clock; this is where that deadline comes from.
 *
 * It lives in memory only. Persisting it would fire storage.onChanged in every
 * content script on every translate, and they sweep on that.
 */
let cache = { status: null, deadlineAt: null, at: 0 };

const snapshot = () => ({
  status: cache.status,
  deadlineAt: cache.deadlineAt,
  at: cache.at,
});

function remember(status) {
  const secs = status.unload_in_secs;
  cache = {
    status,
    // `unload_in_secs` is null when nothing is loaded. A `typeof` test rejects
    // that without also rejecting a genuine 0, which Number() would not.
    deadlineAt: typeof secs === "number" ? Date.now() + secs * 1000 : null,
    at: Date.now(),
  };
  return cache;
}

/* Only fired for changes the popup cannot have caused itself -- a translate
 * finishing in a background tab while it happens to be open. The snapshot rides
 * along so the popup never answers a broadcast with another /status request,
 * which would loop. A rejection just means no popup is listening. */
function announce() {
  browser.runtime
    .sendMessage({ type: "status-changed", ...snapshot() })
    .catch(() => {});
}

/* A finished run is what resets the server's idle timer, so the deadline has to
 * move with it rather than drift until the next /status. */
function noteRunFinished() {
  if (!cache.status) return;
  const idle = cache.status.idle_unload_secs;
  const armed = typeof idle === "number" && idle > 0;
  cache = {
    status: {
      ...cache.status,
      models_loaded: true,
      unload_in_secs: armed ? idle : null,
    },
    deadlineAt: armed ? Date.now() + idle * 1000 : null,
    at: Date.now(),
  };
  announce();
}

/* ------------------------------------------------------------------ requests */

async function fetchStatus(cfg) {
  const res = await fetch(`${base(cfg)}/status`, {
    method: "GET",
    headers: authHeaders(cfg),
  });
  if (!res.ok) throw new Error(`server ${res.status}: ${await detail(res)}`);
  return remember(await res.json());
}

async function warmup(cfg) {
  const res = await fetch(`${base(cfg)}/warmup`, {
    method: "POST",
    headers: authHeaders(cfg),
  });
  if (res.status === 507) {
    const err = new Error(
      (await detail(res)) || "the server has no VRAM free to load its models"
    );
    err.kind = "insufficient_memory";
    throw err;
  }
  if (!res.ok) throw new Error(`server ${res.status}: ${await detail(res)}`);
  return true;
}

async function unloadNow(cfg) {
  const res = await fetch(`${base(cfg)}/unload`, {
    method: "POST",
    headers: authHeaders(cfg),
  });
  if (!res.ok) throw new Error(`server ${res.status}: ${await detail(res)}`);
  // /unload answers with the same body as /status, so one call does both.
  return remember(await res.json());
}

/* Stops the server for good, which is what ends a reading session.
 *
 * The cache is cleared FIRST and the shutdown is only asked for afterwards. The
 * other order loses the guarantee: once the listener closes we have no way to
 * tell a server that stopped from one that was never reachable, so a failed
 * shutdown would leave the pages on disk with the reader believing otherwise.
 * Clearing first can only over-delete, which is the safe direction here.
 *
 * A network error on the way back is success, not failure. The server tears the
 * listener down as it answers, so the response legitimately races the close. */
async function stopServer(cfg) {
  await cacheClear();
  try {
    const res = await fetch(`${base(cfg)}/shutdown`, {
      method: "POST",
      headers: authHeaders(cfg),
    });
    if (!res.ok && res.status !== 0) {
      throw new Error(`server ${res.status}: ${await detail(res)}`);
    }
  } catch (err) {
    /* Distinguish "it refused us" from "it went away mid-reply". A 401 arrives
     * as a real response and is rethrown above; only a transport failure lands
     * here, and that is the expected shape of a server that just stopped. */
    if (err && err.kind) throw err;
  }
  cache = { status: null, deadlineAt: null, at: 0 };
  return true;
}

/* The story this page belongs to, as `config()` resolved it.
 *
 * Nothing is stored: the id is a pure function of the page URL and the per-
 * series reset counter, so it survives the event page being suspended without
 * anyone having to persist it, and two callers cannot drift apart. It does NOT
 * outlive a stopped server, and does not need to -- the server forgot every
 * story when its process ended, so the id simply finds nothing and the next page
 * starts a fresh window. */
async function storyId(cfg) {
  return typeof cfg.storyId === "string" ? cfg.storyId : "";
}

/* Resets the context window for the series the reader is currently looking at:
 * tells the server to forget that story, then bumps a counter so the next page
 * of it computes a different id and starts from nothing.
 *
 * PER SERIES, not per profile, which is the whole point of the change -- a
 * reader who wants to un-learn one book's names should not lose the other book
 * they have open. The button therefore acts on the ACTIVE TAB, and the popup is
 * open over that tab when it is pressed.
 *
 * A counter rather than a fresh UUID, because the id has to stay a pure function
 * of the URL: `cacheLookup` and `translate` both compute it independently and a
 * disagreement is a permanent, symptomless cache miss. The counter is the only
 * part that is state, and it is keyed by series so it moves exactly one story.
 *
 * Told-then-replaced, because the server keys by id -- bumping first would leave
 * the old pairs resident until the process ends. Best-effort on the server side:
 * an unknown id carries nothing anyway, and a server that is down has already
 * forgotten every story, so neither is worth refusing the reset over.
 *
 * NOTE what this costs the cache, because it is not free and it is not new: the
 * story id is inside `settingsFingerprint`, so a reset makes every stored page
 * of that series miss. That is the intended reading -- a reset says "translate
 * this afresh" -- and it is strictly narrower than before, when one reset
 * invalidated every page of every site. */
async function resetStory(cfg) {
  /* The tab the popup is sitting over. `tabs.query` returns a URL here because
   * the manifest holds `<all_urls>`; a window with no active tab, or one whose
   * URL we may not read, falls to the same empty key `storyPageKey` gives any
   * non-URL, which is a real story like any other rather than an error. */
  const [tab] = await browser.tabs
    .query({ active: true, currentWindow: true })
    .catch(() => []);
  const pageUrl = (tab && tab.url) || "";
  const { key } = storyPageKey(pageUrl);

  /* Unconditional, where the old code guarded on a possibly-empty stored id.
   * `storyIdFor` is total -- every URL, and every non-URL, yields an id -- so
   * there is no "no story to clear" state left to test for. */
  const resets = { ...(cfg.storyResets || {}) };
  const body = new FormData();
  body.append("story", storyIdFor(pageUrl, resets));
  await fetch(`${base(cfg)}/story`, {
    method: "POST",
    headers: authHeaders(cfg),
    body,
  }).catch(() => {});

  const count = Number(resets[key]);
  resets[key] = (Number.isInteger(count) && count > 0 ? count : 0) + 1;
  await browser.storage.local.set({ storyResets: resets });
  return { ok: true, storyId: storyIdFor(pageUrl, resets), series: key };
}

/* Installs the series' terms on the server under the story id this request is
 * about to send, before the /translate POST. Whole-glossary replace, POSTed
 * before EVERY translate rather than once per session: the server store is
 * in-memory and this side cannot see a restart, so the idempotent local POST
 * is the self-healing arm. LOAD-BEARING, not best-effort -- the render is
 * cached under `cfg.glossaryFingerprint`, so translating past a failed
 * install would file a glossary-less render behind a glossary key for 24
 * hours (the same shape of bug `keepArtIgnored` exists to stop, one field
 * over). `/story`'s reset POST may shrug; this one must throw, and a server
 * that stored a different count than it was sent is the same bug wearing
 * success.
 *
 * AN EMPTY LIST POSTS A CLEAR RATHER THAN SKIPPING, and that is the heal for
 * a state the adversarial review found: a browser restart wipes
 * storage.session while a standing server keeps the terms under the same
 * URL-derived, restart-stable story id -- skip the POST and every reply
 * carries glossary_terms > 0 against a glossary-less cache key, so the skew
 * guard refuses to cache the series for the LIFE OF THE SERVER PROCESS. The
 * server removes on empty and answers {terms: 0} (glossary.rs::set), so the
 * count check passes unchanged and no empty entry joins the 8-slot store.
 * The cost is one small local POST per translate for glossary-less readers. */
async function installGlossary(cfg, story) {
  const terms = cfg.glossaryTerms || [];
  if (!story) return;
  const body = new FormData();
  body.append("story", story);
  body.append("terms", JSON.stringify(terms));
  let res = await fetch(`${base(cfg)}/glossary`, {
    method: "POST",
    headers: authHeaders(cfg),
    body,
  });
  /* SELF-HEAL ON 401. A stored token is never re-asked while it is
   * non-empty, so without this a stale one strands the reader on a manual
   * popup re-sync. This install runs before BOTH authenticated POSTs (the
   * per-slice translate and the seam), which makes it the session's front
   * door: resync once here and everything behind it inherits the fresh token,
   * because `cfg` is the same object downstream. One retry, never a loop -- a
   * host that answers with the same wrong token falls through to the ordinary
   * thrown 401. */

  if (res.status === 401) {
    const fresh = await tokenResync(cfg);
    if (fresh) {
      res = await fetch(`${base(cfg)}/glossary`, {
        method: "POST",
        headers: authHeaders(cfg),
        body,
      });
    }
  }  if (!res.ok) {
    throw new Error(`glossary install failed: server ${res.status}: ${await detail(res)}`);
  }
  const reply = await res.json().catch(() => ({}));
  if (reply.terms !== terms.length) {
    throw new Error(`glossary install stored ${reply.terms} of ${terms.length} terms`);
  }
}

/* Drop the stored token, ask the native host for the current one, and adopt
 * it IN PLACE on the live cfg so every fetch later in the same request uses
 * it. Returns null when there is no host, no reply, or nothing changed --
 * the cases where a retry would just repeat the 401. */
async function tokenResync(cfg) {
  tokenFromHost = null;
  const reply = await browser.runtime
    .sendNativeMessage(NATIVE_HOST, { command: "token" })
    .catch(() => null);
  if (!reply || !reply.ok || !reply.token || reply.token === cfg.token) return null;
  await browser.storage.local.set({ token: reply.token }).catch(() => {});
  cfg.token = reply.token;
  return cfg;
}

/* Whether the series' terms changed between this request's config() snapshot
 * and now. The wire's `glossary_terms` is a COUNT, so an edit that keeps the
 * count -- fixing one translation while pages are in flight -- is invisible
 * to the skew guard; this reply-time re-read closes that window. A rejected
 * session read answers "drifted", which degrades to shown-not-cached -- the
 * conservative arm. Residual, accepted: an edit-and-revert completing
 * entirely inside one request's flight is still invisible; closing it needs
 * the server to echo a term digest, which is a wire change. */
async function glossaryDrifted(cfg) {
  try {
    const key = glossaryStorageKey(cfg.glossarySeriesKey);
    const stored = await browser.storage.session.get(key);
    const terms = Array.isArray(stored[key]) ? stored[key] : [];
    return glossaryFingerprint(terms) !== (cfg.glossaryFingerprint || "");
  } catch {
    return true;
  }
}

/* The popup editor's round-trip. The popup never computes the series key --
 * story.js is background-only by design -- so both handlers resolve the active
 * tab here, exactly as `resetStory` does. */
async function glossaryGet() {
  const [tab] = await browser.tabs
    .query({ active: true, currentWindow: true })
    .catch(() => []);
  const pageUrl = (tab && tab.url) || "";
  const { key } = storyPageKey(pageUrl);
  const stored = await browser.storage.session.get(glossaryStorageKey(key)).catch(() => ({}));
  const terms = Array.isArray(stored[glossaryStorageKey(key)]) ? stored[glossaryStorageKey(key)] : [];
  return { ok: true, series: key, text: glossaryFormat(terms), count: terms.length };
}

/* `cfg` arrives from the dispatcher's argless `config()`, exactly as
 * `resetStory`'s does -- calling `config(hostOf(pageUrl), pageUrl)` in here
 * would key the OCR latch off the PAGE host, which is the class of bug
 * story-key.test.js's call-site scan exists to refuse. The latch fields do
 * not matter to a /glossary POST; `base`/`authHeaders`/`storyResets` do, and
 * the argless resolve carries all three. */
async function glossarySet(cfg, text) {
  const { terms, errors } = glossaryParse(text);
  if (errors.length) return { ok: false, errors };
  const [tab] = await browser.tabs
    .query({ active: true, currentWindow: true })
    .catch(() => []);
  const pageUrl = (tab && tab.url) || "";
  const { key } = storyPageKey(pageUrl);
  if (terms.length) {
    await browser.storage.session.set({ [glossaryStorageKey(key)]: terms });
  } else {
    await browser.storage.session.remove(glossaryStorageKey(key));
    /* Clearing must also clear the SERVER's copy, or the next translate
     * renders with terms its cache key no longer names. Best-effort is enough
     * HERE because translate()'s skew guard refuses to cache the residue --
     * this POST just makes the common path clean rather than self-healing. */
    const body = new FormData();
    body.append("story", storyIdFor(pageUrl, cfg.storyResets));
    body.append("terms", "[]");
    await fetch(`${base(cfg)}/glossary`, {
      method: "POST",
      headers: authHeaders(cfg),
      body,
    }).catch(() => {});
  }
  return { ok: true, count: terms.length };
}

/* --- The term ledger ----------------------------------------------------
 * Observation, not configuration: every term-shaped (source, translated)
 * pair a translate produces is counted per series, so the popup can ask the
 * reader to arbitrate drift and the pin dialog can offer renderings they
 * have already read. Storage writes are read-modify-write on one key; the
 * per-request region count is small enough that the race window between two
 * concurrent translates costs at worst one lost increment, never a term. */

async function glossaryLedgerFor(seriesKey) {
  const key = glossaryLedgerStorageKey(seriesKey);
  const stored = await browser.storage.session.get(key).catch(() => ({}));
  return stored[key] && typeof stored[key] === "object" ? stored[key] : {};
}

async function glossaryLedgerHarvest(seriesKey, regions) {
  if (!seriesKey || !Array.isArray(regions) || !regions.length) return;
  const ledger = await glossaryLedgerFor(seriesKey);
  const now = Date.now();
  let touched = false;
  for (const region of regions) {
    if (region.refused || !region.source || !region.translated) continue;
    if (!glossaryLedgerTermKey(region.source)) continue;
    glossaryLedgerAdd(ledger, region.source, region.translated, now);
    touched = true;
  }
  if (touched) {
    await browser.storage.session
      .set({ [glossaryLedgerStorageKey(seriesKey)]: ledger })
      .catch(() => {});
  }
}

/* The reply's `terms` array: term-shaped regions only, boxes in the posted
 * image's own pixel space, so the content script can hit-test an alt-click
 * without a round trip. Nothing sentence-shaped rides along -- the filter is
 * the same one the ledger uses, so the two surfaces cannot disagree about
 * what is pinnable. */
function translateTerms(regions) {
  const terms = [];
  for (const region of Array.isArray(regions) ? regions : []) {
    if (region.refused || !region.source || !region.translated) continue;
    if (!glossaryLedgerTermKey(region.source)) continue;
    terms.push({
      x: region.x,
      y: region.y,
      width: region.width,
      height: region.height,
      source: region.source,
      translated: region.translated,
    });
  }
  return terms.length ? terms : null;
}

/* The entry's wording pins -- every lettered region's
 * (source -> translated) pair, compact. A later EDIT APPLY sends them as
 * translation_pins so untouched bubbles keep their wording; a refused or
 * sourceless region never pins, because there is no wording to keep. */
function translatePins(regions) {
  const pins = [];
  for (const region of Array.isArray(regions) ? regions : []) {
    if (region.refused || !region.source || !region.translated) continue;
    pins.push({ s: region.source, t: region.translated });
  }
  return pins.length ? pins : null;
}

/* The pin dialog's choices: the ledger's renderings for one source term on
 * the SENDER's series -- a content-script message, so the active tab would
 * be the wrong key whenever the reader has another window focused. */
async function glossaryPinOptions(pageUrl, source) {
  const { key } = storyPageKey(pageUrl);
  const ledger = await glossaryLedgerFor(key);
  const stored = await browser.storage.session.get(glossaryStorageKey(key)).catch(() => ({}));
  const terms = Array.isArray(stored[glossaryStorageKey(key)]) ? stored[glossaryStorageKey(key)] : [];
  const term = glossaryLedgerTermKey(source) || source;
  const pinned = terms.find((one) => (glossaryLedgerTermKey(one.source) || one.source) === term);
  return {
    ok: true,
    renderings: glossaryLedgerRenderings(ledger, source),
    pinned: pinned ? pinned.translation : null,
  };
}

/* One pin, appended through the SAME parse/validate/store path the popup's
 * Apply uses -- never a raw storage write, or the editor and the pin would
 * drift on validation. Replace-or-append on the normalized source, then
 * delegate the whole reformatted text to glossarySet, which also owns the
 * clear-on-empty and the server POST rules. The pageUrl is the sender's;
 * glossarySet re-resolves the active tab for its own key, which is the same
 * page for a click IN that page, so the two keys agree. */
async function glossaryPin(cfg, pageUrl, source, translation) {
  const term = glossaryLedgerTermKey(source) || (source || "").trim();
  const text = typeof translation === "string" ? translation.trim() : "";
  if (!term || !text) return { ok: false, error: "nothing to pin" };
  const { key } = storyPageKey(pageUrl);
  const stored = await browser.storage.session.get(glossaryStorageKey(key)).catch(() => ({}));
  const terms = Array.isArray(stored[glossaryStorageKey(key)]) ? stored[glossaryStorageKey(key)] : [];
  const kept = terms.filter((one) => (glossaryLedgerTermKey(one.source) || one.source) !== term);
  kept.push({ source: term, translation: text });
  return glossarySet(cfg, glossaryFormat(kept));
}

/* The popup's suggestion list for the active tab's series. */
async function glossarySuggest() {
  const [tab] = await browser.tabs
    .query({ active: true, currentWindow: true })
    .catch(() => []);
  const pageUrl = (tab && tab.url) || "";
  const { key } = storyPageKey(pageUrl);
  const ledger = await glossaryLedgerFor(key);
  const stored = await browser.storage.session.get(glossaryStorageKey(key)).catch(() => ({}));
  const terms = Array.isArray(stored[glossaryStorageKey(key)]) ? stored[glossaryStorageKey(key)] : [];
  return { ok: true, series: key, items: glossaryLedgerSuggestions(ledger, terms) };
}

/* The only thing here that leaves the browser sideways rather than over HTTP.
 *
 * `sendNativeMessage` rejects when no host is registered, which is the ordinary
 * first-run state -- the user has to run Setup.bat once, because registering
 * the host writes to the registry and an extension cannot. That rejection is turned
 * into the instruction rather than surfaced raw.
 *
 * The message carries a command name and nothing else. The host ignores any
 * path or argument by design: a registered native host is reachable by every
 * extension its manifest names, so one that ran what it was told would be a
 * general-purpose process launcher sitting behind a browser. */
const NATIVE_HOST = "birelate.server";

async function startServer() {
  let reply;
  try {
    reply = await browser.runtime.sendNativeMessage(NATIVE_HOST, { command: "start" });
  } catch (err) {
    const message = String((err && err.message) || err);
    /* Firefox words this several ways across versions, so match loosely and
     * fall back to the raw text rather than claiming it is a missing host. */
    if (/no such native application|not found|ENOENT/i.test(message)) {
      throw new Error(
        "no launcher registered - run Setup.bat once, then restart Firefox"
      );
    }
    throw new Error(message);
  }
  if (!reply || !reply.ok) {
    throw new Error((reply && reply.error) || "the launcher did not start the server");
  }
  /* The host starts serve.ps1 with a stable machine-local token and hands it
   * back, so the first click works without a paste. On a fresh start the
   * host's token is authoritative for the server it just launched; when a
   * server was already listening it may be someone else's, so a token the
   * user already holds is kept. */
  if (reply.token) {
    const stored = (await browser.storage.local.get("token")).token;
    if (!reply.already || !stored) {
      await browser.storage.local.set({ token: reply.token });
    }
  }
  return reply;
}

async function ping() {
  const cfg = await config();
  try {
    const res = await fetch(`${base(cfg)}/health`, { method: "GET" });
    return { ok: res.ok, status: res.status };
  } catch (err) {
    return { ok: false, error: String(err.message || err) };
  }
}

/* Named for the user, not for the server: what they need to hear is how much is
 * missing, not which endpoint said no. */
function shortfall(status) {
  const vram = status.vram || {};
  const free = vram.available_bytes;
  const need = status.needed_bytes;
  if (typeof free !== "number" || typeof need !== "number") {
    return "not enough free VRAM on the BireLate server to translate right now";
  }
  return `not enough VRAM: ${gib(free)} free, ${gib(need)} needed`;
}

/* ----------------------------------------------------------------- the cache
 *
 * Two tiers. The content script asks with the URL before it fetches anything;
 * on a miss it fetches as it always did and the bytes land here, where they are
 * hashed. A hit at either tier returns the stored PNG and stops: no /status, no
 * /warmup, no POST, and no noteRunFinished() -- the server did not run, so its
 * idle countdown has not moved and saying otherwise would push the popup's
 * clock out on a translation that never happened.
 */

/* A cache that cannot be read or written is a slower translator, not a broken
 * one, which is why every call site below swallows its own failure. */
const cacheWarn = (what) => (err) => {
  console.warn(`[koharu] cache ${what}:`, String((err && err.message) || err));
  return null;
};

// These come from a number field and from a store that has held whatever an
// earlier version wrote, so neither is trusted to be a usable number.
function cacheLimits(cfg) {
  const entries = Math.floor(Number(cfg.cacheMaxEntries));
  const bytes = Math.floor(Number(cfg.cacheMaxBytes));
  return {
    maxEntries: entries > 0 ? entries : DEFAULTS.cacheMaxEntries,
    maxBytes: bytes > 0 ? bytes : DEFAULTS.cacheMaxBytes,
  };
}

async function cacheLookup({ url }, pageUrl) {
  /* THE HOST IS NOT OPTIONAL HERE, and leaving it out made tier 1 permanently
   * dead on every host the OCR latch had fired on. `config(host)` is what
   * resolves `ocrByHost` and `sourceLanguage`, and BOTH are in the fingerprint --
   * so a lookup without it keys under the DEFAULT engine and `sourceLanguage: ""`
   * while `translate` stored under the latched pair. The `urls` index is queried
   * with `IDBKeyRange.only`, so there is no near-miss and no fallback: every
   * revisit re-fetched the image over the network and re-hashed it, and only
   * tier 2 kept the answer correct.
   *
   * THE PAGE URL IS NOT OPTIONAL EITHER, and for exactly the same reason one
   * layer along: `cfg.storyId` is in the fingerprint too, and it is now derived
   * from the page's series rather than stored. A lookup that resolved it from a
   * different URL than `translate` did -- or from none -- would key under a
   * different story for ever, with no near miss and no fallback. */
  const cfg = await config(hostOf(url), pageUrl);
  const fingerprint = settingsFingerprint(cfg);

  const entry = await cacheGetByUrl(url, fingerprint).catch(cacheWarn("read"));
  // A stored blob whose backing file has gone is a miss, not a failure: the
  // caller can still fetch the image and translate it.
  const dataUrl = entry
    ? await cacheDataUrl(entry.blob).catch(cacheWarn("read"))
    : null;
  if (!dataUrl) return { ok: true, hit: false };

  await cacheTouch(entry.key, url, fingerprint).catch(cacheWarn("touch"));
  await cacheBump("hits").catch(cacheWarn("stats"));
  // The miss report travels with the entry, so a re-read of a partial page says
  // so as loudly as the run that produced it. Entries written before these
  // fields existed simply have no report, which reads as "nothing to say".
  return {
    ok: true,
    hit: true,
    dataUrl,
    missed: entry.missed || null,
    dropped: entry.dropped || null,
    cut: Boolean(entry.cut),
    duplicateIds: entry.duplicateIds || 0,
    outOfRangeIds: entry.outOfRangeIds || 0,
    cutFound: entry.cutFound || 0,
    stillCut: entry.stillCut || 0,
    // Pre-feature entries have no field, which reads as "all inside".
    placementOverflow: entry.placementOverflow || [],
    // Everything the seam needs, so a strip read a second time is still joined
    // without a single request reaching the server for the two slices.
    edges: entry.edges || null,
    seamed: Array.isArray(entry.seamed) ? entry.seamed : [],
    // The box editor's substrate, so a revisit can still be edited.
    // Absent on pre-feature entries, which reads as "nothing to edit yet".
    boxes: entry.boxes || null,
    edits: entry.edits || null,
    terms: entry.terms || null,
  };
}

/* The multipart body, in one place, because the seam MUST send byte-identical
 * model fields to the two slices it joins. `engine::needs_reload` is a
 * structural comparison over the whole PipelineConfig, so one differing field
 * rebuilds the StageRunner with an empty Residency -- which evicts every model
 * including the ~16 GiB translator, to profile each stage alone. That is a cold
 * page at best and a 507 on a contended card at worst. */
function translateForm(cfg, blob, story, joined, seed, edits, pins) {
  const form = new FormData();
  form.append("image", blob, "page.png");
  form.append("target_language", cfg.targetLanguage);
  form.append("ocr", cfg.ocr);
  form.append("inpainting", cfg.inpainting);
  // Always sent, both ways round. The server refuses any spelling but "true" and
  // "false" rather than reading an unexpected value as off, because defaulting
  // would erase the artwork the reader asked to keep.
  /* `segment_context` is deliberately NOT sent. The reader-facing toggle was
   * removed after an A/B measured the ON arm worse on 6 of 7 blind-judged
   * cells -- it lost a character's name in the chapter's headline ability and
   * inverted every recurring title against its settled rendering. The server
   * still accepts the field and still has `--segment-context`, so a harness can
   * A/B it; absent simply falls back to the server's own default, which is
   * what a reader should get. */
  form.append("provider", cfg.provider);
  if (cfg.llm) form.append("llm", cfg.llm);
  /* Only when the latch has something to say. An absent field leaves the
   * server's script rules off, which is the safe arm: a WRONG declaration is
   * the one failure the gate must not have, since it would refuse real dialogue
   * for being in the language it is actually written in. */
  if (cfg.sourceLanguage) form.append("source_language", cfg.sourceLanguage);
  /* The FORMAT axis, caller-stated -- the server never infers it and
   * neither does this field: it carries an explicit pick or the latch's own
   * format evidence, and is absent while neither exists. The server ships it
   * inert for now (parsed, logged, no behavioral consumer), which is why it is
   * NOT in `settingsFingerprint` yet -- see `config()`. */
  if (cfg.profile) form.append("profile", cfg.profile);
  /* Only when a story is running. Absent means "translate this page alone",
   * which is what every request did before stories existed. */
  if (story) form.append("story", story);
  /* ONLY THE SEAM SENDS THIS, and only because the seam is the one caller that
   * built its own image.
   *
   * It tells the server the picture was assembled so that its text ends inside
   * it, which disarms the OCR stage's cross-slice guard. That guard refuses a
   * region touching both edges of its page -- correct on an ordinary slice,
   * where such a region is a fragment of something taller and the translator
   * fabricates from it, and exactly backwards on a run-joined seam, which was
   * cut to hold the whole name. Measured on a webtoon test chapter: the joined
   * column is 0.987 of the seam against a 0.95 threshold, so every
   * run-joined name came back `refused: no OCR engine ever read it` and stayed
   * in Chinese.
   *
   * Sent only when true. An ordinary page must never carry it -- the guard is
   * what stops `1号云守！` becoming `No. 1: Kumomori!`, and a slice really can
   * hold a fragment of something taller. */
  if (joined) {
    form.append("joined", "true");
    // Where the cuts sit inside the composite, so the server's composite
    // rules know which mints genuinely span a cut. Absent on old servers is
    // harmless -- the field is consumed-and-ignored.
    if (Array.isArray(joined) && joined.length) {
      form.append("joined_boundaries", joined.join(","));
    }
  }
  /* ONLY THE RETRY SENDS THIS. It re-seeds the translator's sampler for this
   * one run; absent, the server keeps its fixed constant and an identical
   * request stays byte-identical -- the determinism every measurement arm
   * leans on, which is why an ordinary translate must never carry it. Decimal
   * text; the server refuses garbage with a 400 rather than silently keeping
   * the constant, because a silently-kept constant is a retry button that
   * does nothing. */
  if (seed !== null && seed !== undefined) form.append("seed", String(seed));
  /* The box edits, exactly when non-empty -- `boxeditWireFields` owns
   * the shape and the absent-when-empty rule, and is loaded on both sides so
   * the two cannot drift. */
  for (const [name, value] of Object.entries(boxeditWireFields(edits))) {
    form.append(name, value);
  }
  /* The wording pins, exactly when the caller supplied them -- which
   * translate() does only on an edit apply. Same absent-when-empty rule as the
   * edits. */
  if (Array.isArray(pins) && pins.length) {
    form.append("translation_pins", JSON.stringify(pins));
  }
  /* JSON rather than the PNG, on every request and not only the ones a seam
   * might need. The body carries the region boxes, which is the only way the
   * browser can know a bubble runs off the bottom of a slice -- and it carries
   * strictly more of everything the PNG path put in headers, so this replaces a
   * path rather than adding one. `format` is a FIELD: `?format=json` is silently
   * ignored and answers with a PNG. */
  form.append("format", "json");
  return form;
}

/* The server hands the render back as `data:image/png;base64,...` on the JSON
 * path. Decoded here rather than through `fetch(dataUrl)` because the cache
 * stores a Blob natively and this is the one step between the two -- and because
 * a data: fetch is a network stack round trip for bytes already in memory. */
function pngFromDataUrl(dataUrl) {
  const comma = dataUrl.indexOf(",");
  const binary = atob(dataUrl.slice(comma + 1));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: "image/png" });
}

/* What the PNG path used to say in headers, read off the JSON body instead.
 *
 * One value genuinely changes shape and it is worth being exact about: the
 * header was `<untranslated entities>/<segments submitted>`, and the body has no
 * segment total -- `misses.segments` never leaves the server. The denominator
 * here is the number of regions instead. On an ordinary page each region is one
 * segment and the two agree; where they can differ, this one is the count the
 * reader can actually see on the page in front of them. */
function report(payload) {
  const regions = Array.isArray(payload.regions) ? payload.regions : [];
  const untranslated = Array.isArray(payload.untranslated) ? payload.untranslated : [];
  /* `stages_selected` is the same list `x-birelate-stages` carried, and absence
   * has to mean the same thing it did there: a server too old to report it is
   * indistinguishable from one that reported nothing, and both must fall the
   * same way as "we do not know", never as "the art survived". */
  const selected = Array.isArray(payload.stages_selected) ? payload.stages_selected : [];
  /* The server's two id counters, and they exist because `missed` alone
   * cannot tell the reader WHY a segment is still in its source language. A
   * page that truncated says so through `cut`; a page that simply got no answer
   * says nothing more; and a page where the model answered one segment twice
   * looked, without them, exactly like the second of those. Measured on a real
   * volume: it is not hypothetical and not rare enough to ignore -- three pages
   * of 205, every one of them dropping its LAST segment, with `truncated` false
   * and the reply in input order. Nothing else on the page distinguishes that
   * from an OCR miss.
   *
   * The grammar cannot prevent it: llguidance does not implement `uniqueItems`,
   * so the reply schema pins the NUMBER of entries and not their distinctness.
   * Reporting it is the whole remedy available at this layer.
   *
   * A server too old to send these is indistinguishable from one reporting none,
   * and that is deliberate rather than overlooked: both mean "no cause to name",
   * and a reader can do nothing different with "we do not know". Contrast
   * `stages_selected` above, where the same ambiguity had to fall the other way
   * because something downstream acts on it. */
  const count = (value) => (Number.isInteger(value) && value > 0 ? value : 0);
  /* Regions the server read, erased, and lettered with NOTHING -- a blank patch
   * where a bubble used to be. `missed` cannot carry these and it is structural
   * rather than an oversight: the pipeline writes an unanswered id back holding
   * its source text, so a segment the model skipped renders as Japanese and
   * lands in `untranslated`. It can never come back empty. An empty translation
   * is always an entry the model positively answered with an empty string.
   *
   * The reason this is the loudest of the lot: every other report leaves
   * SOMETHING on the page that points at its own cause. Untranslated text is
   * visibly Japanese; a truncated bubble ends mid-word. A drop leaves no
   * evidence at all -- measured on 59 regions over 22 pages, every one of which
   * answered with `untranslated: []`, `truncated: false` and both id counters
   * zero.
   *
   * What the reader is left looking at depends on the plan, so nothing here
   * asserts it: under the default the box was inpainted before translation ever
   * ran, so it is a blank patch, while "Keep the artwork" runs no inpainting
   * stage and leaves the original Japanese sitting there. Both are the same
   * defect on the wire and only the wording downstream has to care -- see
   * `missNotes` in content.js. */
  const dropped = Array.isArray(payload.dropped) ? payload.dropped : [];
  /* Boxes the detector found at a page edge and the server REFUSED as regions
   * for scoring under its text floor. They are not regions and must never
   * become any: they never reach OCR, are never lettered, are never counted in
   * any of the ratios above, and are never sent back. The seam is the only
   * consumer -- `seamEdges` folds them into the edge summary and nothing else
   * on this page reads them. See `seamEdges` for the column they exist for.
   *
   * ABSENT AND EMPTY FALL THE SAME WAY, unlike `stages_selected` two fields up,
   * and the difference is worth being exact about because that one had to fall
   * the other way. There, "we do not know" and "the art survived" are different
   * answers and something downstream acts on the difference. Here the only act
   * is "consider this box when looking for a cut bubble", so a server too old to
   * send the field and a server that looked and found none produce the same
   * join -- which is precisely the join the seam made before hints existed. */
  const edgeHints = Array.isArray(payload.edge_hints) ? payload.edge_hints : [];
  /* Reads the server marked SUSPECT: a refused watermark or an unread sub-floor
   * box sits close enough to the region that glyphs may be hidden under it and
   * missing from the read -- an occluded glyph can ship inside a fluent
   * translation with nothing to say so. Same absent-and-empty rule as edgeHints:
   * an old server and a clean page both count zero, because the only act here
   * is telling the reader which reads to distrust. */
  const occluded = regions.filter((region) => region && region.occluded_by).length;
  return {
    regions,
    edgeHints,
    missed: untranslated.length ? `${untranslated.length}/${regions.length}` : null,
    dropped: dropped.length ? `${dropped.length}/${regions.length}` : null,
    occluded: occluded ? `${occluded}/${regions.length}` : null,
    
    cut: payload.truncated === true,
    duplicateIds: count(payload.duplicate_ids),
    outOfRangeIds: count(payload.out_of_range_ids),
    /* The cut pair, same absent-falls-to-zero rule as the id counters two
     * fields up: a server too old to send these means "no cause to name".
     * `cutFound` is the translator's own count of segments whose first reply
     * arrived cut mid-sentence; `stillCut` is what SHIPS still cut after the
     * repair -- the reader-visible half, and the only one missNotes voices.
     * Until these, the verdict lived in the server's tracing and a repaired
     * page was indistinguishable from a silently cut one -- an emitted report
     * the browser threw away. */
    cutFound: count(payload.cut_found),
    stillCut: count(payload.still_cut),
    /* Region indices whose placed English left the reader's own
     * placement box -- the 9px floor makes a too-small box spill rather than
     * shrink, and the editor tells the reader to enlarge it. Absent falls to
     * empty like the counters above: an old server means "no cause to name". */
    placementOverflow: Array.isArray(payload.placement_overflow)
      ? payload.placement_overflow
      : [],
    artKept: selected.length > 0 && !selected.includes("inpainting"),
  };
}

async function translate({ bytes, mime, url, width, height, shape, retry, edits }, pageUrl) {
  const host = hostOf(url);
  const cfg = await config(host, pageUrl);
  const fingerprint = settingsFingerprint(cfg);

  /* A RETRY carries no bytes, deliberately. The first live click proved the
   * refetch design wrong: real hosts' urls are routinely dead by retry time
   * (signed urls expire, blob: urls are revoked), and the page is displaying
   * OUR lettering. The entry's stored `source` is the exact stream the first
   * translation hashed, so re-rolling from it provably lands the overwrite
   * on the SAME key -- a refetch could return different bytes and file the
   * re-roll beside the entry it was meant to replace. An entry from before
   * the field, or one that aged out of the 24 hours, cannot be re-rolled:
   * the reader is told to translate the image fresh, which stores it. */
  let data;
  let kind = mime;
  /* The editor's changes this request, or -- on a plain retry -- the entry's
   * own stored edits re-sent whole, so a re-roll never resurrects a box the
   * reader deleted. */
  let effectiveEdits = null;
  /* On an EDIT APPLY, the entry's stored (source -> translated)
   * pairs ride as pins, so an untouched bubble keeps its wording and only
   * the changed box re-translates. A PLAIN retry sends none, or it could
   * never re-roll -- the same asymmetry as the seed below, and keyed on the
   * same `edits` argument. */
  let pins = null;
  if (retry) {
    const prior = await cacheGetByUrl(url, fingerprint).catch(cacheWarn("read"));
    if (!prior || !prior.source) {
      throw new Error(
        "retry needs one fresh translation of this image first — show the original, then translate it again"
      );
    }
    data = new Uint8Array(await prior.source.arrayBuffer());
    kind = prior.source.type || "image/png";
    effectiveEdits = edits || prior.edits || null;
    if (edits) pins = prior.pins || null;
  } else {
    data = new Uint8Array(bytes);
  }

  /* Tier 2. The URL missed, but a host that rotates its paths serves the same
   * picture under a new name, and hashing what was actually fetched catches
   * that. Recording the new URL is what makes the next reload a tier-1 hit.
   *
   * A RETRY exists to REPLACE what is stored, so it must not read it back:
   * this tier is skipped on the way in (the content script already skipped
   * tier 1), and the ordinary cachePut below then overwrites the same key on
   * the way out -- which is what makes the re-roll stick for the next visit.
   * The seed is NOT in the fingerprint, deliberately: a seed in the key would
   * file the re-roll beside the draw it was meant to replace, and every later
   * view would hit the old entry. */
  const srcHash = await sha256Hex(data);
  const key = cacheKey(srcHash, fingerprint);
  const entry = retry ? null : await cacheGetByKey(key).catch(cacheWarn("read"));
  const cached = entry
    ? await cacheDataUrl(entry.blob).catch(cacheWarn("read"))
    : null;

  if (cached) {
    await cacheTouch(key, url, fingerprint).catch(cacheWarn("touch"));
    await cacheBump("hits").catch(cacheWarn("stats"));
    return {
      ok: true,
      dataUrl: cached,
      cached: true,
      missed: entry.missed || null,
      dropped: entry.dropped || null,
      cut: Boolean(entry.cut),
      duplicateIds: entry.duplicateIds || 0,
      outOfRangeIds: entry.outOfRangeIds || 0,
      cutFound: entry.cutFound || 0,
      stillCut: entry.stillCut || 0,
      // Same absent-field rule as its neighbours.
      placementOverflow: entry.placementOverflow || [],
      edges: entry.edges || null,
      seamed: Array.isArray(entry.seamed) ? entry.seamed : [],
      /* Entries from before term pinning have no field and read as nothing to
       * pin -- the same rule `missed` and `dropped` follow for old entries. */
      terms: entry.terms || null,
      // Same absent-field rule for the box editor's pair.
      boxes: entry.boxes || null,
      edits: entry.edits || null,
    };
  }

  /* Pre-flight. A run that starts without the headroom to finish aborts the
   * whole server process, so the image is not worth uploading. A server too old
   * to answer /status is a different matter: only an answer we understood is
   * allowed to stop us. */
  let status = null;
  try {
    ({ status } = await fetchStatus(cfg));
  } catch (err) {
    console.warn("[koharu] /status unavailable:", String(err.message || err));
  }

  if (status && status.sufficient === false) {
    const err = new Error(shortfall(status));
    err.kind = "insufficient_memory";
    throw err;
  }

  if (status && status.models_loaded === false) {
    // Loading the models takes far longer than the request that needs them, so
    // /warmup returns at once and the POST below simply waits its turn on the
    // server's GPU permit.
    await warmup(cfg);
  }

  const blob = new Blob([data], { type: kind });
  const story = await storyId(cfg);
  await installGlossary(cfg, story);

  /* The whole reason a retry reaches the server at all: the translator's
   * sampler seed is a fixed constant, so an identical request is answered
   * byte-identically at any temperature -- the re-roll
   * has to SAY it wants a different draw. Random rather than sequential, so
   * two retries of the same page do not replay each other. An EDIT APPLY
   * deliberately carries no seed: the correction should be deterministic,
   * and the retry button stays the only re-roll. */
  const seed = retry && !edits ? crypto.getRandomValues(new Uint32Array(1))[0] : null;

  const started = Date.now();
  const res = await fetch(`${base(cfg)}/translate`, {
    method: "POST",
    body: translateForm(cfg, blob, story, null, seed, effectiveEdits, pins),
    headers: authHeaders(cfg),
  });

  if (res.status === 507) {
    // The server's own pre-flight beat ours -- something else took the VRAM
    // between our /status and this POST.
    const err = new Error((await detail(res)) || "the server is out of VRAM");
    err.kind = "insufficient_memory";
    throw err;
  }
  if (!res.ok) throw new Error(`server ${res.status}: ${await detail(res)}`);

  const payload = await res.json();
  const dataUrl = typeof payload.image === "string" ? payload.image : "";
  if (!dataUrl.startsWith("data:image/")) {
    throw new Error("the server answered without an image");
  }
  const out = pngFromDataUrl(dataUrl);

  /* The miss report, which dropping was the last place a token-cap overflow
   * could still hide. The server does the whole job -- scales the budget to the
   * page, notices `finish_reason == Length`, counts the regions it never filled
   * in -- and the browser used to throw the answer away, so an overflowing page
   * rendered surplus Japanese bubbles that look exactly like an OCR miss. That
   * is the failure the server's miss report exists to remove, one layer up.
   * Kept on the cache entry too: a warning that vanishes on the second view of
   * the same page is barely a warning.
   *
   * `artKept` answers a different question with the same care. An unknown
   * multipart field is consumed in silence by design, so a server built before
   * skip_inpainting existed answers 200 with a fully inpainted page and nothing
   * in the body says so -- and that erased artwork would then be filed under a
   * keep-the-art fingerprint for 24 hours. The skew is a normal state: the
   * extension and the server are updated independently. */
  const { regions, edgeHints, missed, dropped, cut, duplicateIds, outOfRangeIds,
          cutFound, stillCut, placementOverflow, artKept } =
    report(payload);
  const keepArtIgnored = Boolean(cfg.keepArt) && !artKept;
  /* A render made under different terms than the key claims must be shown and
   * never remembered -- `keepArtIgnored`'s rule, one field over. The wire's
   * `glossary_terms` exists exactly so a stored arm is self-describing; an
   * absent field coerces to 0 because a story-less request and an older server
   * both mean "no glossary ran". This is the guard that closes every desync
   * the install path cannot see: a cleared editor with the server still
   * holding terms, or a server restarted between the install and this reply. */
  const glossarySkew =
    (payload.glossary_terms || 0) !== (cfg.glossaryTerms || []).length ||
    (await glossaryDrifted(cfg));
  if (glossarySkew) {
    console.warn(
      `[koharu] glossary skew: server ran ${payload.glossary_terms || 0} terms, ` +
        `this series has ${(cfg.glossaryTerms || []).length}; page shown, not cached`
    );
  }

  /* Fed the OCR output, not the pixels, and that ordering is forced rather than
   * chosen. `ocr.rs` says so in the file itself: a crop the reader cannot
   * resolve does not fail, it returns fluent invented prose -- so there is
   * nothing in an image to route on beforehand, and the first page or two of a
   * new host is necessarily read by whichever engine is currently selected.
   * That cost is bounded and paid once per host; the alternative is a reader who
   * never learns their chapter was fabricated.
   *
   * Deliberately not awaited into the response path: a storage write must not
   * delay the page the reader is waiting for, and nothing below depends on it. */
  void noteScript(host, regions, shape).catch(() => {});
  /* The term ledger's harvest, same rules as noteScript: fire and forget,
   * never in the reply's path. Keyed on the reset-invariant series key so a
   * story reset does not orphan the observations. */
  void glossaryLedgerHarvest(cfg.glossarySeriesKey, regions).catch(() => {});
  const terms = translateTerms(regions);
  const storedPins = translatePins(regions);
  /* The box editor's substrate: this reply's region rectangles,
   * compact, kept with the entry so a cache hit can still open the editor.
   * Rects only -- the strings stay on the wire reply, the editor shows boxes. */
  const boxes = regions.map((region) => ({
    x: region.x,
    y: region.y,
    width: region.width,
    height: region.height,
  }));

  /* Remember which image host served this page host, so the popup -- which can
   * only know the PAGE host -- can find the latch, which is keyed on the IMAGE
   * host. Display plumbing, not a decision input: `resolveSite` never reads it.
   * Written only when it changes, and not awaited for the same reason
   * `noteScript` is not: a storage write must not delay the page. */
  const seenPageHost = hostOf(pageUrl);
  if (seenPageHost && host && (cfg.imageHostForPage || {})[seenPageHost] !== host) {
    void browser.storage.local
      .set({ imageHostForPage: { ...(cfg.imageHostForPage || {}), [seenPageHost]: host } })
      .catch(() => {});
  }

  /* The seam's whole input, computed once here rather than in the content
   * script: seam.js is loaded on both sides precisely so this is the same
   * function, and the box list is far cheaper to summarise before it is stored
   * than to ship whole.
   *
   * Computed even when the reader has the join switched OFF, which looks like
   * waste and is not. `joinSlices` is deliberately absent from the cache
   * fingerprint, so an entry written with it off is the very entry a later read
   * with it on will hit -- and there is no way to recover the boxes from a
   * stored blob. Gating this on the setting made "turning it on takes effect on
   * the next view of a cached strip" false for a full 24 hours, silently, and
   * pointed at the wrong field (`seamed`) while doing it. It is arithmetic over
   * a list already in hand.
   *
   * `edgeHints` rides in HERE and stops here, which is the whole of their
   * plumbing. Folding them into the summary rather than storing them beside it
   * is what makes them survive the 24-hour cache for free: `edges` is already
   * written by `cachePut`, read back by both `cacheLookup` and the tier-2 hit
   * above, and handed to the content script under one name. A parallel field
   * would be a second thing to remember at five call sites, and forgetting it at
   * any one of them makes the join fire once on the live reply and never again
   * on the revisit -- which is the failure this whole path exists to avoid. */
  /* The white-continuation profile is measured HERE because this is
   * the one place that has both the flag and the bytes -- seam.js is pixel-free
   * by contract, and the content script's canvas is taintable while these bytes
   * are ours. Folded into `edges` so it survives the 24-hour cache exactly the
   * way `edgeHints` do. Fails open to null: a probe error must not cost the
   * translation, only the trigger. */
  const white =
    cfg.touchJoin && width > 0 && height > 0
      ? await edgeWhiteProfile(data, width, height).catch(() => null)
      : null;
  const edges =
    width > 0 && height > 0 ? seamEdges(regions, width, height, edgeHints, white) : null;

  /* Not filed at all when the art was supposed to survive and did not. The
   * fingerprint records what was *asked for*, so storing this page would put an
   * erased image behind a keep-the-art key for 24 hours -- and every later view
   * would tier-1 hit it without ever reaching the server again, which is exactly
   * the shape of bug the fingerprint exists to prevent. Showing the page is
   * still right; remembering it is not.
   *
   * The write is awaited, not fired and forgotten: this page is an event page,
   * and a write still in flight when it is suspended has its transaction aborted
   * rather than committed. Firefox keeps it alive while a message reply is
   * outstanding, so the write has to finish before that reply resolves. */
  if (!keepArtIgnored && !glossarySkew) {
    await cachePut({
      srcHash,
      fingerprint,
      url,
      blob: out,
      /* The bytes this very request uploaded -- srcHash's own preimage --
       * kept so a later RETRY has an honest byte source after the host's url
       * has died. The same blob object the upload used, type included. */
      source: blob,
      ms: Date.now() - started,
      missed,
      dropped,
      cut,
      duplicateIds,
      outOfRangeIds,
      cutFound,
      stillCut,
      placementOverflow,
      edges,
      terms,
      pins: storedPins,
      boxes,
      edits: effectiveEdits,
    }).catch(cacheWarn("write"));
    await cacheEnforceCaps(cacheLimits(cfg)).catch(cacheWarn("trim"));
  }

  // Counted here rather than at the lookup, so that the two counters the popup
  // shows are hits against translations that actually happened.
  await cacheBump("misses").catch(cacheWarn("stats"));

  noteRunFinished();
  /* `seamed` is empty by construction on a page that has only just been
   * translated, and is sent anyway so the content script reads one shape from
   * both a fresh run and a cache hit. */
  return {
    ok: true, dataUrl, missed, dropped, cut, duplicateIds, outOfRangeIds,
    cutFound, stillCut, placementOverflow,
    keepArtIgnored, edges, seamed: [], terms, boxes, edits: effectiveEdits,
  };
}

/* ------------------------------------------------------------- the seam
 *
 * The bytes and the pixels, all of them. seam.js decided WHAT to cut and where
 * to paint it back; this cuts, sends, and paints.
 *
 * It lives here rather than in the content script for one reason that outranks
 * the rest: the correction has to end up in the CACHE, or a strip re-read
 * tomorrow is two tier-1 hits that never reach the server and show the split
 * bubble again. The cache is this script's, and baking the join into the two
 * stored pages is also what answers "should a seam be cached?" -- it is not, it
 * has no URL of its own and two fresh entries sharing one URL member resolve by
 * index order rather than recency. What gets cached is still exactly "the
 * translation of image X".
 */

/* A background page in Firefox MV3 (`"background": {"scripts": [...]}`, not a
 * service worker) is a hidden DOM document, so `document.createElement` is the
 * same API content.js already uses for its canvas -- and unlike the content
 * script there is no cross-origin taint to defeat, because every pixel here
 * arrives as bytes we were handed. Preferred over OffscreenCanvas for exactly
 * that reason: if this file ever became a service worker the failure is a loud
 * TypeError caught below, not a subtly different rendering path. */
const seamReady = () =>
  typeof document !== "undefined" && typeof createImageBitmap === "function";

function seamCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

const seamPng = (canvas) =>
  new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("could not encode the joined page"))),
      "image/png"
    );
  });

/* How deep the balloon-white continues into a slice from each of
 * its cut edges, measured per column and summarised per `STEP`-column bucket.
 *
 * The trigger's question is "does the neighbour's edge band continue this
 * balloon >= 60 px?", and the numbers here serve exactly that: `DEPTH` bounds
 * the walk at 160 px because the census's deepest lobe worth distinguishing is
 * ">= 300" and the gate is at 60 -- a lobe deeper than 160 reads as 160 and
 * still fires. `WHITE` at 235 is balloon white against JPEG noise: webtoon
 * balloons are near-pure white, and a threshold much lower starts reading
 * pale art as balloon. Each bucket keeps its MINIMUM column depth, so a
 * speedline crossing the lobe shortens the evidence rather than averaging
 * away -- conservative in the direction that avoids a false join. */
const SEAM_TOUCH_PROBE_DEPTH = 160;
const SEAM_TOUCH_PROBE_STEP = 8;
const SEAM_TOUCH_PROBE_WHITE = 235;

async function edgeWhiteProfile(bytes, width, height) {
  if (!seamReady()) return null;
  const bitmap = await createImageBitmap(new Blob([bytes]));
  try {
    const depth = Math.min(SEAM_TOUCH_PROBE_DEPTH, bitmap.height);
    if (depth <= 0 || bitmap.width <= 0) return null;
    const probe = (fromTop) => {
      const canvas = seamCanvas(bitmap.width, depth);
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(bitmap, 0, fromTop ? 0 : depth - bitmap.height);
      const pixels = context.getImageData(0, 0, bitmap.width, depth).data;
      const buckets = Math.ceil(bitmap.width / SEAM_TOUCH_PROBE_STEP);
      const out = new Array(buckets);
      for (let bucket = 0; bucket < buckets; bucket += 1) {
        let least = depth;
        const from = bucket * SEAM_TOUCH_PROBE_STEP;
        const to = Math.min(bitmap.width, from + SEAM_TOUCH_PROBE_STEP);
        for (let x = from; x < to; x += 1) {
          let run = 0;
          while (run < depth) {
            const row = fromTop ? run : depth - 1 - run;
            const i = (row * bitmap.width + x) * 4;
            const luminance =
              0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
            if (luminance < SEAM_TOUCH_PROBE_WHITE) break;
            run += 1;
          }
          if (run < least) least = run;
        }
        out[bucket] = least;
      }
      return out;
    };
    return { top: probe(true), bottom: probe(false), step: SEAM_TOUCH_PROBE_STEP };
  } finally {
    if (typeof bitmap.close === "function") bitmap.close();
  }
}

/* Cut the tail of the head slice, every middle slice whole, and the head of the
 * tail slice, stacked with no gap. Zero overlap is not an approximation: the
 * host sliced one strip, so the last row of k and the first row of k+1 really
 * are adjacent rows of one picture, and every community stitching tool
 * concatenates them directly.
 *
 * N bands rather than two, because a skill name drawn down the side of a panel
 * crosses slices that hold neither its beginning nor its end, and no pairwise
 * join can ever make it whole. `plan.slices[i].offset` is where band i starts in
 * the seam, so the loop is the two-slice draw generalised and not a new one. */
async function seamImage(plan, bitmaps) {
  const canvas = seamCanvas(plan.width, plan.height);
  const context = canvas.getContext("2d");
  for (let i = 0; i < plan.slices.length; i += 1) {
    const band = plan.slices[i];
    context.drawImage(
      bitmaps[i],
      0,
      band.y,
      plan.width,
      band.height,
      0,
      band.offset,
      plan.width,
      band.height
    );
  }
  return await seamPng(canvas);
}

/* Paint the rejoined bubbles into one slice's stored translation.
 *
 * Only the rectangles seam.js chose, never the whole band: the seam is its own
 * page, so the server's size coherence solved a different base for it and every
 * bubble that merely fell inside the crop was re-lettered at a size belonging to
 * the seam's population. Dragging those back would make a correct page
 * inconsistent in order to fix one bubble. */
async function seamPaint(entry, seamBitmap, parts, expect) {
  const page = await createImageBitmap(entry.blob);
  try {
    if (page.width !== expect.width || page.height !== expect.pageHeight) return null;
    const canvas = seamCanvas(page.width, page.height);
    const context = canvas.getContext("2d");
    context.drawImage(page, 0, 0);
    for (const part of parts) {
      context.drawImage(
        seamBitmap,
        part.source.x,
        part.source.y,
        part.source.width,
        part.source.height,
        part.target.x,
        part.target.y,
        part.target.width,
        part.target.height
      );
    }
    return await seamPng(canvas);
  } finally {
    page.close();
  }
}

async function seamJoin({ plan, slices }, pageUrl) {
  /* Same host, same reason, and here it silently killed the whole feature.
   * Without it the two `cacheGetByUrl` calls below miss on a latched host, the
   * `skipped: "uncached"` arm returns BEFORE the POST, and `performSeam` reads
   * that `ok` reply as the boundary answered and writes `SEAM_SETTLED` -- so
   * every boundary on every strip settles as answered-nothing, after paying two
   * image re-fetches and two multi-MB marshals for a reply that is thrown away.
   *
   * It also has to be the host and not "drop the fields from the fingerprint":
   * `cfg` builds the POST body too (`translateForm`), and the seam MUST send
   * byte-identical model fields or `needs_reload` rebuilds the StageRunner with
   * an empty Residency. Fixing only the lookup would have turned a dead seam
   * into a pipeline reload per boundary.
   *
   * The page URL rides along for the same reason and with the same teeth: the
   * seam's two `cacheGetByUrl` calls must key under the story the two slices
   * were stored under, and its own POST must be prompted with that story or the
   * rejoined bubble is translated with no memory of the page it belongs to. */
  const sources = Array.isArray(slices) ? slices : [];
  const cfg = await config(hostOf(sources[0] && sources[0].url), pageUrl);
  // The fold of the old `joinSlices` checkbox: the seam runs unless the SITE
  // resolves to paged manga. Undeclared keeps the old default.
  if (cfg.profile === "manga") return { ok: true, joined: 0, skipped: "manga-profile" };
  if (!seamReady()) throw new Error("this browser cannot compose images in the background");
  if (!plan || !plan.width || !plan.height) throw new Error("no seam to join");
  if (!Array.isArray(plan.slices) || plan.slices.length !== sources.length) {
    throw new Error("the seam plan and its slices disagree");
  }

  const fingerprint = settingsFingerprint(cfg);
  /* `index` is which band of the SEAM a slice supplied; `edges` are which of
   * that slice's OWN edges the join repaired, and the two are deliberately not
   * the same thing -- see `seamRunEdges`, which owns that rule because the
   * content script's in-memory copy of it has to give the same answer. */
  const sides = sources.map((source, index) => ({
    index,
    edges: seamRunEdges(index, sources.length),
    url: source.url,
    bytes: source.bytes,
    mime: source.mime,
    page: plan.slices[index],
  }));

  /* Both translations have to still be on disk, because they are what the join
   * is painted into. A page the reader asked to keep the artwork on is never
   * filed at all (see `keepArtIgnored`), and an entry can have aged out between
   * the two slices, so this is an ordinary outcome rather than an error. */
  for (const side of sides) {
    side.entry = await cacheGetByUrl(side.url, fingerprint);
    if (!side.entry) return { ok: true, joined: 0, skipped: "uncached" };
  }

  const bitmaps = [];
  try {
    for (const side of sides) {
      side.source = await createImageBitmap(
        new Blob([new Uint8Array(side.bytes)], { type: side.mime })
      );
      bitmaps.push(side.source);
      /* The plan was measured against the picture the content script had. If the
       * bytes decode to anything else -- a host that reuses one URL for two
       * pictures, an entry from a differently sized render -- every offset below
       * is wrong by an unknown amount, and a silently misplaced paste is worse
       * than no join at all. */
      if (side.source.width !== plan.width || side.source.height !== side.page.pageHeight) {
        return { ok: true, joined: 0, skipped: "mismatch" };
      }
    }

    const blob = await seamImage(
      plan,
      sides.map((side) => side.source)
    );
    const story = await storyId(cfg);
    await installGlossary(cfg, story);
    const res = await fetch(`${base(cfg)}/translate`, {
      method: "POST",
      /* THE PLAN'S BOUNDARIES HERE AND NOWHERE ELSE. This is the one caller
       * that assembled its own image out of a run of slices, so it is the one
       * caller entitled to say the text ends inside it -- and to say where the
       * cuts sit, which is what the server's composite rules key on. A
       * non-empty array is truthy, so old readers of this parameter keep
       * reading "joined". */
      body: translateForm(cfg, blob, story, plan.boundaries || true),
      headers: authHeaders(cfg),
    });

    if (res.status === 507) {
      const err = new Error((await detail(res)) || "the server is out of VRAM");
      err.kind = "insufficient_memory";
      throw err;
    }
    /* A seam is one or two bubbles, so it is far more likely than a whole page to
     * come back with nothing translated -- the server answers that with a 502 and
     * discards the render. Both slices already read correctly, so this is a join
     * that did not happen rather than a page that failed. */
    if (!res.ok) throw new Error(`server ${res.status}: ${await detail(res)}`);

    const payload = await res.json();
    const image = typeof payload.image === "string" ? payload.image : "";
    if (!image.startsWith("data:image/")) throw new Error("the server answered without an image");
    /* Counted BEFORE the skew check: the server did run a full translate
     * whether or not the join is painted, and the idle-unload accounting must
     * see it -- translate()'s skew path keeps its own noteRunFinished for the
     * same reason. */
    noteRunFinished();
    /* The joined render must be the same arm as the two cached slices it is
     * about to be pasted into. A skew here would paint wrong-arm pixels into
     * two 24-hour entries, which is worse than no join at all -- skip, exactly
     * like "mismatch" above; the boundary can try again on a later view. The
     * drift re-read closes the same-count edit window, as in translate(). */
    if (
      (payload.glossary_terms || 0) !== (cfg.glossaryTerms || []).length ||
      (await glossaryDrifted(cfg))
    ) {
      return { ok: true, joined: 0, skipped: "glossary-skew" };
    }

    /* WHAT THE JOIN PRODUCED, not merely that one happened.
     *
     * `performSeam` already logs "joined" with the crop size, and that was the
     * whole of the seam's observability -- enough to prove the feature is alive
     * and useless for every question about the result. Without this, a joined
     * region lettered too small could only be investigated by rebuilding the
     * composite by hand and rendering it offline, which measured a DIFFERENT
     * number from the browser's.
     *
     * Colour is here beside size because they share an input -- `infer_text_color`
     * and `mask_font_size` both read the region's segmentation mask -- so a wrong
     * mask moves the two together, and seeing only one of them hides which. */
    const seamRegions = Array.isArray(payload.regions) ? payload.regions : [];
    /* The seam's composite is where a cut display column becomes whole, which
     * makes its regions the best term-shaped strings a session produces --
     * harvest them into the same ledger the per-slice path feeds. */
    void glossaryLedgerHarvest(cfg.glossarySeriesKey, seamRegions).catch(() => {});
    const seamLayers = Array.isArray(payload.rendered_text) ? payload.rendered_text : [];
    /* `layout_warnings` rides along because it is the ONLY field that separates
     * "the fitter chose this size" from "the fitter gave up and used the floor".
     * `run_auto` returns at `minimum` on a failed search without logging anything
     * (`layout.rs`, three silent `run_with_size(text, minimum)` paths), so the
     * warning's `actual_width`/`actual_height` against the box is the only way to
     * tell an honest floor from a missed size. */
    console.debug(
      `[birelate] seam result ${plan.width}x${plan.height} slices=${plan.slices.length} ` +
        `head.y=${plan.slices[0].y} cuts=${plan.boundaries.join(",")} ` +
        `regions=${seamRegions.length} layers=${seamLayers.length} ` +
        `layout_warnings=${JSON.stringify(payload.layout_warnings || null)}`,
      seamRegions.map((region, index) => {
        const layer = seamLayers.find((one) => one.region === index) || null;
        return {
          i: index,
          box: [region.x, region.y, region.width, region.height].map((v) => Math.round(v)),
          color: region.color,
          region_font: region.font_size,
          placed_font: layer ? layer.font_size : null,
          chars: layer ? layer.chars : null,
          refused: region.refused || null,
          occluded_by: region.occluded_by || null,
        };
      })
    );

    const rects = seamCompositeRects(plan, seamRegions);
    /* For diagnosing TOUCHES-class DOUBLING: the joined lettering painted
     * beside the un-erased tail of the slice-local pass, and nothing in the
     * existing trace shows the PAIR BOXES the plan carried or the RECTS the
     * painter used -- the two facts that decide whether the paint rect
     * under-covered the local lettering (and why). Debug-only, no behaviour
     * change; the fit extents ride so the rect can be judged against what
     * the local pass actually lettered. */
    console.debug(
      `[birelate] seam paint pairs=${JSON.stringify(
        (plan.pairs || []).map((pair) => ({
          cut: pair.cut,
          top: [pair.top.x, pair.top.y, pair.top.width, pair.top.height].map(Math.round),
          bottom: [pair.bottom.x, pair.bottom.y, pair.bottom.width, pair.bottom.height].map(
            Math.round
          ),
        }))
      )} rects=${JSON.stringify(
        rects.map((one) => ({
          box: [one.rect.x, one.rect.y, one.rect.width, one.rect.height],
          parts: one.parts.map((part) => ({
            slice: part.slice,
            src: [part.source.x, part.source.y, part.source.width, part.source.height],
            dst: [part.target.x, part.target.y, part.target.width, part.target.height],
          })),
        }))
      )} sideEdges=${JSON.stringify(
        sides.map((side) => {
          const edges = (side.entry && side.entry.edges) || {};
          const brief = (boxes) =>
            (boxes || []).map((box) =>
              [box.x, box.y, box.width, box.height].map(Math.round)
            );
          return { top: brief(edges.top), bottom: brief(edges.bottom) };
        })
      )}`
    );
    const seamBitmap = rects.length ? await createImageBitmap(pngFromDataUrl(image)) : null;
    if (seamBitmap) bitmaps.push(seamBitmap);

    const out = [];
    for (const side of sides) {
      const parts = seamBitmap
        ? rects.flatMap((rect) => rect.parts.filter((part) => part.slice === side.index))
        : [];
      /* Re-read rather than reuse the entry fetched before the POST. A middle
       * slice of a strip belongs to TWO boundaries, so it can be joined at its
       * top and at its bottom, and the two joins each read, paint and write the
       * whole page: painting the second onto bytes read before the first was
       * stored would silently discard the first. The content script serialises
       * its own joins for the same reason; this covers a second tab reading the
       * same strip, which it cannot. */
      const entry = parts.length
        ? await cacheGetByUrl(side.url, fingerprint).catch(cacheWarn("read"))
        : null;
      const painted = entry
        ? await seamPaint(entry, seamBitmap, parts, {
            width: plan.width,
            pageHeight: side.page.pageHeight,
          })
        : null;
      /* Marked even when nothing was painted on this side, and even when the
       * seam produced no rectangle at all. The GPU run has happened; leaving the
       * edge unmarked would buy the same answer again on every revisit for as
       * long as the two entries live. Written back before the reply resolves,
       * not after: Firefox keeps this event page alive only while a message
       * reply is outstanding, and a transaction still in flight when it is
       * suspended is aborted rather than committed. */
      await cacheMarkSeamed(side.entry.key, painted, side.edges).catch(cacheWarn("seam"));
      /* Positional, so a slice that was painted nothing still occupies its own
       * place in the reply. The content script pairs this against the run it
       * sent; a compacted list would silently shift every join onto the wrong
       * picture the first time a middle slice produced no rectangle. */
      out[side.index] = painted ? await cacheDataUrl(painted) : null;
    }

    /* A re-encoded page is not the byte-for-byte size of the one the server
     * sent, so a join moves the store's total. Every other write path trims
     * after itself; this one has to as well, or the configured ceiling is a
     * ceiling everything but the seam respects. */
    await cacheEnforceCaps(cacheLimits(cfg)).catch(cacheWarn("trim"));
    return { ok: true, joined: rects.length, painted: out };
  } finally {
    for (const bitmap of bitmaps) bitmap.close();
  }
}

/* ------------------------------------------------------------- cache expiry */

const SWEEP_ALARM = "khr-cache-sweep";
const SWEEP_MINUTES = 60;

/* Hourly rather than daily: a daily sweep leaves an entry alive for up to 48
 * hours, which is not what "kept for 24 hours" means.
 *
 * Guarded by a get(), because creating an alarm that already exists restarts
 * its period -- and this page wakes on every single translate, so an unguarded
 * create() would reset the hour long before it ever elapsed. Both fields are
 * passed: with periodInMinutes alone the first firing time is unspecified. */
async function ensureSweepAlarm() {
  const existing = await browser.alarms.get(SWEEP_ALARM);
  if (existing) return;
  browser.alarms.create(SWEEP_ALARM, {
    delayInMinutes: SWEEP_MINUTES,
    periodInMinutes: SWEEP_MINUTES,
  });
}

async function sweepCache() {
  const { expired, evicted } = await cacheSweep(cacheLimits(await config()));
  if (expired || evicted) {
    console.info(`[koharu] cache: ${expired} expired, ${evicted} over cap`);
  }
}

const runSweep = () => sweepCache().catch(cacheWarn("sweep"));

/* -------------------------------------------------------------- frame gating
 *
 * The content script runs in every frame now, and a subframe's own hostname is
 * not necessarily what the user ticked: the popup keys the enable list on the
 * *tab's* host, so a reader served from a sibling subdomain never matches
 * itself. Only this script can see a frame's tab URL, so it is the one authority
 * on the question and the two hostname computations cannot drift apart.
 *
 * It is a narrow inheritance, not a blanket one -- see pageEnabled.
 */
const hostOf = (url) => {
  try {
    return new URL(url).hostname;
  } catch {
    // Not a URL, or one this extension holds no permission to read.
    return "";
  }
};

/* Which page a message came from, for `storyIdFor`.
 *
 * THE TAB'S URL FIRST, and the frame's only as a fallback. The content script
 * runs in every frame, and a reader embedded in an iframe is still reading the
 * TAB's series -- keying on the frame would give an embedded reader its own
 * story, and a site that serves its reader from a sibling subdomain a second one
 * again. `pageEnabled` already treats the tab URL as the one authority for the
 * same reason; this keeps the two computations on the same input.
 *
 * A sender with no tab at all is the popup, which has no page and asks nothing
 * that needs one. */
const pageUrlOf = (sender) =>
  (sender && sender.tab && sender.tab.url) || (sender && sender.url) || "";

/* No public-suffix list is available to an extension, so "same site" here is the
 * conservative shape rather than the registrable domain: identical hosts, or one
 * a subdomain of the other. `cdn.example.com` inside `example.com` passes;
 * `doubleclick.net` inside `example.com` does not. `a.example.co.uk` inside
 * `b.example.co.uk` does not either, which errs the safe way -- a reader that
 * needs it can still be named in the list, or covered by "every page". */
function sameSite(a, b) {
  if (!a || !b) return false;
  return a === b || a.endsWith(`.${b}`) || b.endsWith(`.${a}`);
}

async function pageEnabled(sender) {
  const cfg = await config();
  const host = hostOf(sender && sender.tab && sender.tab.url);
  if (cfg.enabledEverywhere) return { ok: true, enabled: true, host };

  const hosts = Array.isArray(cfg.enabledHosts) ? cfg.enabledHosts : [];
  const frame = hostOf(sender && sender.url);

  /* The tab being enabled is not on its own enough. The content script runs in
   * every third-party iframe on the page now, and answering "yes, the tab is
   * enabled" made each of those a fully enabled instance: auto mode swept the
   * ad, uploaded a 300x250 banner to the GPU server on every chapter page, and
   * a floating ad frame painted a second translate button over the reader. Only
   * a frame belonging to the same site as the tab inherits the tab's answer. */
  const enabled =
    Boolean(host) && hosts.includes(host) && sameSite(frame, host);

  return { ok: true, enabled, host, frame };
}

/* ------------------------------------------------------------------- routing */

browser.runtime.onMessage.addListener((message, sender) => {
  if (!message || typeof message.type !== "string") return;

  // Returning the promise is what makes these async listeners work in Firefox.
  switch (message.type) {
    /* `pageUrlOf(sender)` and not a field on the message: the story key decides
     * both what the server is prompted with and part of the cache key, and only
     * this script can see a frame's tab URL. Reading it here is what stops the
     * three call sites below computing it three ways. */
    case "translate":
      return translate(message, pageUrlOf(sender)).catch(fail);

    /* Tier 1, asked before the content script fetches anything. A failure here
     * is answered as an ordinary error and the caller reads it as a miss, so a
     * dead database costs a refetch rather than a translation. */
    case "cache-lookup":
      return cacheLookup(message, pageUrlOf(sender)).catch(fail);

    /* The two slices are already translated and already on screen when this
     * arrives, so a failure here costs the join and nothing else -- which is why
     * the content script logs it rather than raising it at the reader. */
    case "seam":
      return seamJoin(message, pageUrlOf(sender)).catch(fail);

    case "page-enabled":
      return pageEnabled(sender).catch(fail);

    case "ping":
      return ping();

    case "status":
      return config()
        .then(fetchStatus)
        .then(() => ({ ok: true, ...snapshot() }))
        .catch(fail);

    case "warmup":
      return config()
        .then(async (cfg) => {
          await warmup(cfg);
          // Re-read rather than guess: /warmup is fire-and-forget, so only the
          // server can say whether the models are up yet.
          await fetchStatus(cfg);
          return { ok: true, ...snapshot() };
        })
        .catch(fail);

    case "unload":
      return config()
        .then(unloadNow)
        .then(() => ({ ok: true, ...snapshot() }))
        .catch(fail);

    case "start-server":
      return startServer()
        .then((reply) => ({ ok: true, already: Boolean(reply.already) }))
        .catch(fail);

    case "stop-server":
      return config()
        .then(stopServer)
        .then(() => ({ ok: true, stopped: true }))
        .catch(fail);

    /* One action, three spellings. "story-new" and "story-end" are what older
     * popups send, and both meant "stop carrying what this story has learned"
     * -- which is exactly a reset now that there is no off state. Keeping them
     * costs two lines and stops a popup left open across an update from
     * silently doing nothing. */
    case "story-reset":
    case "story-new":
    case "story-end":
      return config().then(resetStory).catch(fail);

    /* The glossary editor's round-trip. Resolved here and not in the popup
     * for the same reason story-reset is: only this script computes the
     * series key, and story.js stays background-only. */
    case "glossary-get":
      return glossaryGet().catch(fail);

    case "glossary-set":
      return config()
        .then((cfg) => glossarySet(cfg, message.text))
        .catch(fail);

    /* The term ledger's three surfaces. Suggest is the popup's (active tab);
     * the pin pair is the content script's, keyed on the SENDER's page because a
     * click can land while another window is current. */
    case "glossary-suggest":
      return glossarySuggest().catch(fail);

    case "glossary-pin-options":
      return glossaryPinOptions(pageUrlOf(sender), message.source).catch(fail);

    case "glossary-pin":
      return config()
        .then((cfg) => glossaryPin(cfg, pageUrlOf(sender), message.source, message.translation))
        .catch(fail);

    /* "status-changed" is our own broadcast coming back round, and
     * "diagnostic-report" is a content script answering the popup directly.
     * Neither is ours to handle. */
    default:
      return;
  }
});

browser.runtime.onInstalled.addListener(() => {
  browser.contextMenus.create({
    id: "khr-translate",
    title: "Translate image with BireLate",
    contexts: ["image"],
  });

  /* The re-roll, beside the translate. Shown on every image because the menu
   * cannot know which ones hold a translation; the content script degrades a
   * retry on an untranslated image to an ordinary translate. */
  browser.contextMenus.create({
    id: "khr-retry",
    title: "Retry translation (new draw)",
    contexts: ["image"],
  });

  /* The history list used to live in storage.local, which every content script
   * reads in full at document_idle. The cache replaced it; this drops what an
   * earlier version left behind.
   *
   * `storyId` joins it, and the migration decision is: DROP IT, do not adopt it.
   *
   * It is a UUID that was shared by every title the profile ever opened, so
   * there is no series it could honestly be handed to -- adopting it for
   * whichever series happened to be read next would carry the other titles'
   * terminology into that one, which is the exact defect being removed. The
   * pairs it names are server-side and die with the server process anyway, so
   * nothing is lost by orphaning it.
   *
   * COSMETIC, not load-bearing: `config()` overwrites `cfg.storyId` on every
   * call, so a profile that never sees this event is already correct. That
   * matters because `onInstalled` does not fire on an ordinary browser start. */
  browser.storage.local.remove(["history", "historyLimit", "storyId"]).catch(() => {});

  /* Not redundant with onStartup: a temporary add-on loaded from about:debugging
   * never sees onStartup, so without this the sweep would never once run during
   * development and the expiry would look broken. */
  ensureSweepAlarm().catch(cacheWarn("alarm"));
  runSweep();
});

/* Alarms do not fire while the browser is closed, so a machine left off
 * overnight comes back with a store nothing has swept. */
browser.runtime.onStartup.addListener(() => {
  ensureSweepAlarm().catch(cacheWarn("alarm"));
  runSweep();
});

browser.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === SWEEP_ALARM) runSweep();
});

browser.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "khr-translate" && info.menuItemId !== "khr-retry") return;
  if (!tab || typeof tab.id !== "number") return;

  /* The content script owns DOM access, so it resolves the element -- but it is
   * told which one. Forwarding this as "translate everything visible" turned a
   * single named picture into a page-wide job the user never asked for, and did
   * nothing at all on a host the extension was not switched on for, which is
   * every host on a fresh install.
   *
   * A tab with no content script -- about:, the PDF viewer, an injection that
   * failed -- rejects the send. That is a fact about the tab, not an error.
   *
   * Addressed to one frame. The script runs in all of them now, and an
   * untargeted send would have every frame that does not hold the image raise
   * "could not find that image" at the user. `targeted` tells the frame that
   * did receive it that it is allowed to complain. */
  const targeted = typeof info.frameId === "number";
  const payload = {
    type: info.menuItemId === "khr-retry" ? "retry-one" : "translate-one",
    srcUrl: info.srcUrl,
    targeted,
  };
  try {
    // Branched rather than passing `undefined` as the options argument, which
    // the schema validator is entitled to read as a bad value rather than none.
    if (targeted) {
      await browser.tabs.sendMessage(tab.id, payload, { frameId: info.frameId });
    } else {
      await browser.tabs.sendMessage(tab.id, payload);
    }
  } catch (err) {
    console.warn("[koharu] no content script in that tab:", String(err.message || err));
  }
});

/* Belt and braces for a profile whose alarm was lost some way neither of the
 * events above covers. It costs one guarded query per wake of this page, and it
 * deliberately does not sweep -- that would run on every translate. */
ensureSweepAlarm().catch(cacheWarn("alarm"));
