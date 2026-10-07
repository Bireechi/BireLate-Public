//! The multipart body `background.js` sends.
//!
//! Fields arrive in wire order and `llm` is omitted entirely when the popup's
//! field is blank, so the reader is a plain loop over whatever turns up rather
//! than a positional decode.

use axum::{
    body::Bytes,
    extract::{
        Multipart,
        multipart::{MultipartError, MultipartRejection},
    },
    http::StatusCode,
};

use crate::error::{ApiError, clip};

/// How much of a multipart parser's own complaint is echoed. Its wording is not
/// ours to control, so it is treated as untrusted text.
const ECHO: usize = 60;

#[derive(Debug, Default)]
pub struct TranslateForm {
    pub image: Option<Bytes>,
    pub target_language: Option<String>,
    pub ocr: Option<String>,
    pub inpainting: Option<String>,
    pub provider: Option<String>,
    /// Legitimately absent: the extension only appends it when the popup's
    /// Model box is non-empty.
    pub llm: Option<String>,
    /// Whether to leave the artwork alone -- run detection, OCR and translation
    /// but not inpainting, so the English is drawn over the untouched page.
    ///
    /// Deliberately a flag and not a stage list. Koharu enforces stage
    /// prerequisites only *among the stages you selected*, so an arbitrary
    /// subset is accepted in silence: ask for detection and OCR without
    /// translation and the renderer's `fallback_to_source_text` re-typesets the
    /// Japanese over itself in Arial. One boolean makes the single useful
    /// non-full operation reachable and nothing else.
    pub skip_inpainting: Option<String>,
    /// Whether to describe each segment to the translator -- what KIND of text
    /// it is, and whether the artwork strikes it through.
    ///
    /// Per request, and unlike `skip_inpainting` it CANNOT be free of a reload:
    /// stage choice lives on koharu's `Request`, but this has to be inside
    /// `TranslationConfig` to reach the prompt at all, so flipping it costs the
    /// same full pipeline reload an `ocr` or `inpainting` change costs. That is
    /// the accepted trade for a session-level switch and the wrong one for a
    /// per-page one -- the popup presents it as a setting, not a per-page
    /// control.
    ///
    /// Absent falls back to the process default (`--segment-context`), so a
    /// caller that predates the toggle behaves exactly as it did.
    pub segment_context: Option<String>,
    /// Diagnostic: run detection and inpainting only, and render the cleaned
    /// page with no text on it at all.
    ///
    /// The point is to make the *mask* visible in pixels. Inpainting quality is
    /// dominated by which pixels it is told to erase, and today that question can
    /// only be answered by looking at a finished page with English already
    /// painted over the evidence. Deliberately excludes OCR: with no `SourceText`
    /// in the scene the renderer's `fallback_to_source_text` has nothing to fall
    /// back to, so the page comes out genuinely blank rather than re-typeset in
    /// Japanese.
    pub clean_only: Option<String>,
    /// Run DETECTION and nothing else: the boxes and the sub-floor hints, no OCR,
    /// no translation, no eraser.
    ///
    /// For the extension's seam lookahead. The seam plan is computed from GEOMETRY
    /// ALONE — `extension/seam.js`'s `seamEdges` reads x/y/width/height and never
    /// touches `source` or `translated` — and both `regions` and `edge_hints` are
    /// produced by detection. So a client that wants to decide every boundary in a
    /// strip *before* the reader reaches it needs this stage and no other.
    ///
    /// **This is the exception the `skip_inpainting` note above argues against,
    /// and it survives that argument for one reason.** The danger of an arbitrary
    /// stage subset is the renderer re-typesetting Japanese over itself; that
    /// needs a `SourceText`, which needs OCR. Detection alone produces none, so
    /// this lands in the same safe corner `clean_only` already occupies. Anything
    /// that adds OCR to this plan walks straight into the trap.
    ///
    /// Still a flag rather than a stage list, for the rest of that note's reason:
    /// the useful non-full operations are enumerable, and an arbitrary subset is
    /// accepted in silence.
    pub detect_only: Option<String>,
    /// This image was ASSEMBLED so that its text does not continue past its own
    /// edges -- the extension's webtoon seam sends it, and nothing else should.
    ///
    /// It disarms the OCR stage's cross-slice guard and nothing else. That guard
    /// refuses a region reaching both edges of its page, because on an ordinary
    /// slice such a region is a fragment of something taller and the translator
    /// fabricates from it. A seam is cut from a RUN of slices precisely to hold
    /// such a name whole, so there the identical shape means the join worked --
    /// and measured on one test chapter the joined column is 0.987 of the seam,
    /// past the 0.95 threshold, so every run-joined name was refused and left
    /// untranslated.
    ///
    /// Carried on the REQUEST rather than the pipeline config, alongside `story`
    /// and for the same reason: a config that changes per page makes every page
    /// a cold start, and a seam is already a third pipeline run per boundary.
    pub joined: Option<String>,
    /// Where the cuts sit inside a joined image, comma-separated pixels from
    /// its top -- `"972"` or `"300,900"`. Optional beside `joined` so a caller
    /// that predates it stays valid; rules that need a cut's position simply
    /// do not fire without it.
    pub joined_boundaries: Option<String>,
    /// What language the page is in, as a BCP-47 tag -- `ja`, `ko`, `zh`.
    ///
    /// **The pipeline cannot answer this and the caller can.** Before the stamp
    /// echoed this field, every region came back stamped `ja-JP` -- on a mixed test
    /// corpus, all 302 Chinese and Korean regions included -- because `ocr.rs`
    /// used that as its unconditional fallback. The stamp now echoes
    /// this field's resolved declaration back as `regions[].source_language`
    /// (`ja-JP` only when nothing is declared), which is an echo, not an
    /// answer: the extension already measures the kana share per host to choose
    /// an OCR engine, so it knows; this field is how it says so.
    ///
    /// `labels.rs`'s script rules read it to make themselves *stricter*
    /// (absent, they do not fire at all), and the erase veto's script arm takes
    /// it through `SourceScript::language()`.
    pub source_language: Option<String>,
    /// What KIND of publication the site serves -- `manga` (paged) or `webtoon`
    /// (vertical strip). The format axis of the profile split, distinct from
    /// `source_language` on purpose: a Japanese webtoon has been measured at
    /// exactly the 690px width of a Korean manhwa test corpus, so the one thing
    /// a page's shape may never imply is its script, and vice versa.
    ///
    /// CALLER-STATED, like `joined` and unlike everything the pipeline infers:
    /// the extension resolves it from the reader's per-site pick or its own
    /// accumulated shape evidence, and the server never guesses. Absent means
    /// undeclared, which is every caller that predates the field.
    ///
    /// INERT for now: parsed, validated and logged, with no behavioral
    /// consumer. It exists so format-conditional levers (at least one threshold
    /// has been shown not to work as a single global constant across formats) hang
    /// off one declared axis instead of each re-deriving the fork. The change
    /// that gives it a consumer must add it to the extension's
    /// `settingsFingerprint` in the same patch -- `cache.js` says why.
    pub profile: Option<String>,
    /// Opt-in for the richer response shape.
    pub format: Option<String>,
    /// The reader's declared story, minted by the popup's "New story" button.
    ///
    /// Absent means "translate this page alone", which is what every request
    /// did before stories existed.
    pub story: Option<String>,
    /// `/glossary` only: the JSON term array to install for `story`. Parsed by
    /// `glossary::parse_terms`; `/translate` never reads it.
    pub terms: Option<String>,
    /// The box editor's additions: a JSON array of
    /// `{x, y, width, height}` boxes, in the source image's own pixel space,
    /// drawn by the READER over text the detector missed. Admitted into the
    /// settled detection list as asserted text regions; a box holding no ink
    /// is refused pipeline-side and logged. Absent -- every caller but the
    /// editor -- leaves the detector's answer whole.
    pub regions_add: Option<String>,
    /// The box editor's deletions: same JSON shape. Every detection whose box
    /// center lies inside one of these is dropped after the settle pass and
    /// BEFORE the erase masks are written, which is what genuinely un-erases
    /// the artwork underneath. Refused-on-garbage like every declared field.
    pub regions_remove: Option<String>,
    /// The box editor's placement overrides: a JSON list of flat
    /// 8-key entries -- a target region rect plus a `place_*` hard boundary
    /// the region's English must fit inside. Consumed at the RENDER seam
    /// (`placement.rs`), never by detection: the target is matched
    /// centre-inside against the settled regions after the gates run, so a
    /// placement can aim at a detector region without replacing its geometry.
    /// Refused-on-garbage; absent leaves every fit box the renderer's own.
    pub regions_place: Option<String>,
    /// The edit apply's wording pins: a JSON list of
    /// `{s, t}` source/translation pairs from the entry's stored regions.
    /// After the pipeline runs, any region whose OCR source matches a pin's
    /// `s` byte-identically has its translation REPLACED by `t`
    /// (`pins.rs`, before the scene gates) -- so an untouched bubble keeps
    /// its wording across an edit apply and only the changed box re-rolls.
    /// The extension sends this on an EDIT APPLY only, never a plain retry,
    /// which exists to re-roll. Refused-on-garbage; absent pins nothing.
    pub translation_pins: Option<String>,
    /// The translator's sampler seed for this one request -- the extension's
    /// "Retry translation" button, the one caller that WANTS a different draw.
    ///
    /// Absent -- every other caller -- keeps the fixed upstream constant, and
    /// with it the byte-identical replay every measurement depends on. Parsed
    /// as a decimal `u32` and refused on garbage like every declared field
    /// here: a seed silently falling back to the constant would make the retry
    /// button a no-op with no symptom.
    pub seed: Option<String>,
}

pub async fn read_form(
    mut multipart: Multipart,
    max_upload_bytes: usize,
) -> Result<TranslateForm, ApiError> {
    let mut form = TranslateForm::default();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| multipart_error(&error, max_upload_bytes))?
    {
        // `name()` borrows the field while `bytes()`/`text()` consume it, so the
        // name has to be copied out first.
        let Some(name) = field.name().map(str::to_owned) else {
            drop(field.bytes().await);
            continue;
        };
        // The only field that is not text, so it is taken before the table.
        if name == "image" {
            form.image = Some(
                field
                    .bytes()
                    .await
                    .map_err(|error| multipart_error(&error, max_upload_bytes))?,
            );
            continue;
        }
        // One table, not two. This used to be a `|`-list of accepted names and a
        // second `match` mapping each to its slot, whose fallback arm was
        // `_ => &mut form.format` -- so a name added to the first list and missed
        // in the second landed silently in `format`, turning a JSON request into
        // a PNG and losing the value with no error. A single table cannot drift.
        let slot = match name.as_str() {
            "target_language" => Some(&mut form.target_language),
            "ocr" => Some(&mut form.ocr),
            "inpainting" => Some(&mut form.inpainting),
            "skip_inpainting" => Some(&mut form.skip_inpainting),
            "segment_context" => Some(&mut form.segment_context),
            "clean_only" => Some(&mut form.clean_only),
            "detect_only" => Some(&mut form.detect_only),
            "provider" => Some(&mut form.provider),
            "llm" => Some(&mut form.llm),
            "format" => Some(&mut form.format),
            "story" => Some(&mut form.story),
            "terms" => Some(&mut form.terms),
            "source_language" => Some(&mut form.source_language),
            "profile" => Some(&mut form.profile),
            "joined" => Some(&mut form.joined),
            "joined_boundaries" => Some(&mut form.joined_boundaries),
            "seed" => Some(&mut form.seed),
            "regions_add" => Some(&mut form.regions_add),
            "regions_remove" => Some(&mut form.regions_remove),
            "regions_place" => Some(&mut form.regions_place),
            "translation_pins" => Some(&mut form.translation_pins),
            _ => None,
        };
        let Some(slot) = slot else {
            // Unknown fields are consumed rather than left in the stream, which
            // would stall the next `next_field`.
            drop(field.bytes().await);
            continue;
        };
        *slot = Some(
            field
                .text()
                .await
                .map_err(|error| multipart_error(&error, max_upload_bytes))?,
        );
    }
    Ok(form)
}

/// The body limit does not surface as an extractor rejection. `Multipart` only
/// wraps the limited body, so the breach arrives here instead -- as a 413 from a
/// *field* read, several fields into the request.
fn multipart_error(error: &MultipartError, max_upload_bytes: usize) -> ApiError {
    let status = error.status();
    if status == StatusCode::PAYLOAD_TOO_LARGE {
        return ApiError::new(
            status,
            format!(
                "upload exceeds --max-upload-bytes ({max_upload_bytes}); \
                 shrink the image or raise the limit"
            ),
        );
    }
    ApiError::new(
        status,
        format!("bad multipart body: {}", clip(&error.body_text(), ECHO)),
    )
}

/// The extractor itself only ever rejects an unusable boundary, which is a 400.
pub fn rejection_error(rejection: &MultipartRejection) -> ApiError {
    ApiError::new(
        rejection.status(),
        format!("bad multipart body: {}", clip(&rejection.body_text(), ECHO)),
    )
}

/// Whether the caller opted into the JSON shape.
///
/// A bare `fetch` sends `Accept: */*`, and the extension's success path expects
/// raw image bytes, so a wildcard must not count as a request for JSON.
#[must_use]
pub fn wants_json(format: Option<&str>, accept: Option<&str>) -> bool {
    if format.is_some_and(|value| value.eq_ignore_ascii_case("json")) {
        return true;
    }
    accept.is_some_and(|accept| {
        accept.split(',').any(|entry| {
            entry
                .split(';')
                .next()
                .is_some_and(|media| media.trim().eq_ignore_ascii_case("application/json"))
        })
    })
}

#[cfg(test)]
mod tests {
    use axum::{
        Router,
        body::Body,
        extract::{DefaultBodyLimit, FromRequest as _},
        http::{Request, header},
        routing::post,
    };
    use tower::ServiceExt as _;

    use super::*;
    use crate::error::BUDGET;

    const BOUNDARY: &str = "XbireLateX";

    fn body_from(fields: &[(&str, &str)]) -> String {
        let mut body = String::new();
        for (name, value) in fields {
            body.push_str(&format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            ));
        }
        body.push_str(&format!("--{BOUNDARY}--\r\n"));
        body
    }

    async fn parse(fields: &[(&str, &str)]) -> TranslateForm {
        let request = Request::builder()
            .method("POST")
            .uri("/translate")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(Body::from(body_from(fields)))
            .unwrap();
        let multipart = Multipart::from_request(request, &()).await.unwrap();
        read_form(multipart, 1024).await.unwrap()
    }

    #[tokio::test]
    async fn field_order_does_not_matter() {
        let forward = parse(&[
            ("image", "PNG"),
            ("target_language", "en-US"),
            ("ocr", "manga-ocr"),
        ])
        .await;
        let reversed = parse(&[
            ("ocr", "manga-ocr"),
            ("target_language", "en-US"),
            ("image", "PNG"),
        ])
        .await;
        assert_eq!(forward.ocr, reversed.ocr);
        assert_eq!(forward.target_language, reversed.target_language);
        assert_eq!(forward.image.as_deref(), reversed.image.as_deref());
    }

    #[tokio::test]
    async fn a_missing_llm_is_absent_not_an_error() {
        let form = parse(&[("provider", "ollama")]).await;
        assert_eq!(form.provider.as_deref(), Some("ollama"));
        assert!(form.llm.is_none());
    }

    #[tokio::test]
    async fn a_text_field_never_leaks_into_the_format_slot() {
        // The reader used to accept names in a `|`-list and map them to slots in
        // a second match whose fallback arm was `format`. A name in the first
        // list and missing from the second silently became the response shape --
        // so this is a guard against future drift rather than a regression test:
        // `skip_inpainting` was in neither list before, and would have been
        // dropped outright. Sending every text field at once pins each to its
        // own slot, which the single table now makes structurally true.
        let form = parse(&[
            ("target_language", "en-US"),
            ("ocr", "baberu-ocr"),
            ("inpainting", "lama"),
            ("skip_inpainting", "true"),
            ("clean_only", "false"),
            ("detect_only", "true"),
            ("provider", "local"),
            ("llm", "gemma4-31b-it"),
            ("profile", "webtoon"),
            ("format", "json"),
            ("regions_place", "[]"),
            ("translation_pins", "[]"),
        ])
        .await;
        assert_eq!(form.target_language.as_deref(), Some("en-US"));
        assert_eq!(form.ocr.as_deref(), Some("baberu-ocr"));
        assert_eq!(form.inpainting.as_deref(), Some("lama"));
        assert_eq!(form.skip_inpainting.as_deref(), Some("true"));
        assert_eq!(form.clean_only.as_deref(), Some("false"));
        // Added here and not only in its own test, because this is the
        // guard against exactly the drift that once dropped `skip_inpainting`: a
        // field present in the struct and missing from the dispatch table is
        // consumed in silence, and the caller cannot tell.
        assert_eq!(form.detect_only.as_deref(), Some("true"));
        assert_eq!(form.provider.as_deref(), Some("local"));
        assert_eq!(form.llm.as_deref(), Some("gemma4-31b-it"));
        // `profile` sits one row above `format` in the table, and the
        // failure this guards is the file's own: a declared axis landing in the
        // response-shape slot would turn a webtoon declaration into a PNG.
        assert_eq!(form.profile.as_deref(), Some("webtoon"));
        assert_eq!(form.format.as_deref(), Some("json"));
        // Same drift guard: a placement list landing in `format` (or
        // consumed in silence) would letter English somewhere the reader did
        // not choose, with nothing on the wire saying so.
        assert_eq!(form.regions_place.as_deref(), Some("[]"));
        // A pins list consumed in silence re-rolls every bubble the
        // reader thought they were keeping.
        assert_eq!(form.translation_pins.as_deref(), Some("[]"));
    }

    #[tokio::test]
    async fn an_absent_profile_is_undeclared_not_an_error() {
        // Every caller that predates the field. The format axis must cost old
        // callers nothing, exactly like `joined_boundaries` beside it.
        let form = parse(&[("ocr", "baberu-ocr")]).await;
        assert_eq!(form.profile, None);
    }

    #[tokio::test]
    async fn a_later_field_does_not_undo_the_requested_format() {
        // The failure the old shape invited: `format` first, then a field whose
        // name reached the outer list without reaching the inner one. Both wrote
        // the same slot, the last won, and a caller that asked for JSON got a
        // PNG. Nothing in the current reader can do that; this keeps it so.
        let form = parse(&[("format", "json"), ("skip_inpainting", "true")]).await;
        assert!(wants_json(form.format.as_deref(), None));
        assert_eq!(form.skip_inpainting.as_deref(), Some("true"));
    }

    #[tokio::test]
    async fn unknown_fields_are_consumed_and_ignored() {
        let form = parse(&[("mystery", "value"), ("ocr", "baberu-ocr")]).await;
        assert_eq!(form.ocr.as_deref(), Some("baberu-ocr"));
    }

    #[tokio::test]
    async fn an_oversized_body_is_a_413_not_a_500() {
        const LIMIT: usize = 64;
        let app = Router::new().route(
            "/",
            post(|multipart: Multipart| async move {
                match read_form(multipart, LIMIT).await {
                    Ok(_) => (StatusCode::OK, String::new()),
                    Err(error) => (error.status, error.message),
                }
            })
            .route_layer(DefaultBodyLimit::max(LIMIT)),
        );
        let body = body_from(&[("image", &"P".repeat(4096))]);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/")
                    .header(
                        header::CONTENT_TYPE,
                        format!("multipart/form-data; boundary={BOUNDARY}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let message = String::from_utf8(
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(message.contains("--max-upload-bytes"), "{message}");
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
    }

    #[tokio::test]
    async fn an_unusable_boundary_is_a_plain_text_400() {
        let request = Request::builder()
            .method("POST")
            .uri("/translate")
            // No boundary parameter, so the extractor cannot even start.
            .header(header::CONTENT_TYPE, "multipart/form-data")
            .body(Body::from("whatever"))
            .unwrap();
        let Err(rejection) = Multipart::from_request(request, &()).await else {
            panic!("a boundary-less multipart body should be rejected");
        };
        let error = rejection_error(&rejection);
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(error.message.len() <= BUDGET);
    }

    #[test]
    fn json_is_opt_in() {
        assert!(wants_json(Some("json"), None));
        assert!(wants_json(None, Some("application/json")));
        assert!(wants_json(
            None,
            Some("text/html, application/json;q=0.9")
        ));
        assert!(!wants_json(None, Some("*/*")));
        assert!(!wants_json(None, None));
    }
}
