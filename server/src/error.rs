//! Plain-text errors sized for the extension's reader.
//!
//! `background.js` does `detail.slice(0, 120)` on a failed response, so the
//! actionable part of every message has to land inside the first 120 bytes.
//! Anything a request can influence is clipped before it is reflected back.

use std::any::Any;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

/// Everything past this is invisible to the extension.
pub const BUDGET: usize = 120;

/// What to pass `clip` when the clipped text is the *whole* message.
///
/// `clip` appends its ellipsis after filling the limit, so it returns up to
/// `limit + 3` bytes. Passing `BUDGET` directly overshoots by three and the
/// extension then truncates mid-ellipsis.
pub const CLIP_BUDGET: usize = BUDGET - 3;

pub const MISSING_IMAGE: &str = "missing the \"image\" field in the multipart body";
pub const UNSUPPORTED_IMAGE: &str = "unsupported image: send png, jpeg or webp";
pub const UNAUTHORIZED: &str =
    "unauthorized: X-Koharu-Token missing or wrong; paste the server's token into the extension popup";
pub const BAD_HOST: &str =
    "rejected Host header: expected a loopback host; refusing a possible DNS-rebinding request";
pub const RENDER_TASK_FAILED: &str = "the translation task did not finish; check the server log";
pub const UNLOAD_BUSY: &str =
    "server busy: the GPU is in use; the models are freed once the current work finishes";
/// `panic_any` carries no text, and neither does a payload of any other type.
pub const UNPRINTABLE_PANIC: &str = "the payload was not a string; check the server log";

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // `String`'s IntoResponse already sets text/plain; charset=utf-8, which
        // is what the extension's `res.text()` path wants.
        (self.status, self.message).into_response()
    }
}

/// The innermost cause behind an error, and nothing else.
///
/// `PipelineError`'s `Display` stops at the outermost `anyhow` context, and
/// `stage_error` sets that context to a bare `"{model} failed"` /
/// `"failed to load {model}"` -- 34 bytes naming only what the request already
/// asked for. Whatever actually went wrong -- a cuDNN version mismatch,
/// connection refused, no space on the device -- is reachable only through
/// `source`, so the head is dropped and the root goes on the wire. The whole
/// chain goes to the log instead; see `chain`.
#[must_use]
pub fn root_cause(error: &(dyn std::error::Error + 'static)) -> String {
    let mut root: &(dyn std::error::Error + 'static) = error;
    while let Some(source) = root.source() {
        root = source;
    }
    let cause = strip_panic_scaffolding(&root.to_string());
    if cause.is_empty() {
        error.to_string()
    } else {
        cause
    }
}

/// Every link in a source chain, oldest last. For the log, never the wire: a
/// maintainer needs all of them, and the extension has room for one.
#[must_use]
pub fn chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut links = vec![error.to_string()];
    let mut current: &(dyn std::error::Error + 'static) = error;
    while let Some(source) = current.source() {
        links.push(source.to_string());
        current = source;
    }
    links.join(" <- ")
}

/// The two `Result` unwrap prefixes. Their `Option` counterparts are left alone
/// because they are followed by nothing -- stripping one yields an empty string,
/// which is worse than the boilerplate.
const UNWRAP_PREFIXES: [&str; 2] = [
    "called `Result::unwrap()` on an `Err` value: ",
    "called `Result::expect()` on an `Err` value: ",
];

/// Peels the wrappers a panic accretes on its way into an error message.
///
/// The observed cuDNN failure wore three at once. koharu runs every ONNX/torch
/// stage in `spawn_blocking`, so the panic reaches Rust as tokio's
/// `task <id> panicked with message "..."`; the panic itself is
/// `koharu-torch`'s `.unwrap()`, so it reads `called `Result::unwrap()` on an
/// `Err` value: ...`; and the value printed is the `{:?}` of a one-field
/// newtype, `Torch("...")`. That is 77 bytes of scaffolding in front of the
/// first letter of CUDNN, and with koharu's own 61-byte head in front of that
/// the extension's entire 120-byte window held nothing but punctuation.
#[must_use]
pub fn strip_panic_scaffolding(text: &str) -> String {
    // Bounded rather than looped to a fixed point: three layers are what the
    // real failure carries, and a bound makes termination a property of this
    // function rather than of the three parsers below.
    let mut current = text.trim().to_owned();
    for _ in 0..4 {
        let peeled = peel_one(&current);
        if peeled == current {
            break;
        }
        current = peeled;
    }
    current
}

fn peel_one(text: &str) -> String {
    if let Some(inner) = inside_tokio_join(text) {
        return inner;
    }
    for prefix in UNWRAP_PREFIXES {
        if let Some(rest) = text.strip_prefix(prefix) {
            return rest.to_owned();
        }
    }
    inside_debug_newtype(text).unwrap_or_else(|| text.to_owned())
}

/// tokio's `JoinError` for a panicked task, whose task id means nothing outside
/// the runtime that issued it.
fn inside_tokio_join(text: &str) -> Option<String> {
    let (_, message) = text
        .strip_prefix("task ")?
        .split_once(" panicked with message ")?;
    unquote_debug(message)
}

/// `{:?}` of a single-field tuple struct, which is the shape every
/// `koharu-torch` error arrives in. The type's name says less than the stage
/// prefix the caller has already added.
fn inside_debug_newtype(text: &str) -> Option<String> {
    let (name, rest) = text.split_once('(')?;
    if name.is_empty()
        || !name
            .chars()
            .all(|character| character.is_alphanumeric() || character == '_' || character == ':')
    {
        return None;
    }
    unquote_debug(rest.strip_suffix(')')?)
}

/// Undoes `{:?}` on a string. `None` for anything that was not quoted, so a
/// caller can tell "not this shape" from "this shape, now unwrapped".
fn unquote_debug(text: &str) -> Option<String> {
    let inner = text.strip_prefix('"')?.strip_suffix('"')?;
    let mut unquoted = String::with_capacity(inner.len());
    let mut characters = inner.chars();
    while let Some(character) = characters.next() {
        if character != '\\' {
            unquoted.push(character);
            continue;
        }
        // The escaped character is pushed as itself unless it names a control
        // one, which covers both `\"` and `\\`. A trailing lone backslash cannot
        // come out of `{:?}`, but it must not silently vanish either.
        match characters.next() {
            Some('n') => unquoted.push('\n'),
            Some('t') => unquoted.push('\t'),
            Some('r') => unquoted.push('\r'),
            Some('0') => unquoted.push('\0'),
            Some(other) => unquoted.push(other),
            None => unquoted.push('\\'),
        }
    }
    Some(unquoted)
}

/// The message inside a panic payload.
///
/// `panic!` with a literal stores a `&'static str` and every formatted one a
/// `String`; `panic_any` stores whatever it was given, which carries no text at
/// all -- hence the constant rather than a `{:?}` of unknown length.
#[must_use]
pub fn panic_text(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        return strip_panic_scaffolding(text);
    }
    if let Some(text) = payload.downcast_ref::<String>() {
        return strip_panic_scaffolding(text);
    }
    UNPRINTABLE_PANIC.to_owned()
}

/// Shortens untrusted text before it is echoed into an error.
///
/// The budget is in *bytes* because the 120-character window the extension
/// applies is really a byte window once a hostile page sends multi-byte input,
/// but characters are never split -- `&value[..limit]` would panic.
#[must_use]
pub fn clip(value: &str, limit: usize) -> String {
    let mut clipped = String::with_capacity(limit + 3);
    for character in value.chars() {
        if clipped.len() + character.len_utf8() > limit {
            clipped.push_str("...");
            return clipped;
        }
        clipped.push(character);
    }
    clipped
}

#[cfg(test)]
mod tests {
    // `source` is a trait method, and `Layer` below is only reachable as one.
    use std::error::Error as _;

    use super::*;

    #[test]
    fn every_constant_message_fits_the_extensions_window() {
        for message in [
            MISSING_IMAGE,
            UNSUPPORTED_IMAGE,
            UNAUTHORIZED,
            BAD_HOST,
            RENDER_TASK_FAILED,
            UNLOAD_BUSY,
            UNPRINTABLE_PANIC,
            crate::guard::BAD_ORIGIN_SUFFIX,
        ] {
            assert!(
                message.len() <= BUDGET,
                "{} bytes: {message}",
                message.len()
            );
        }
    }

    #[test]
    fn responses_are_plain_text_with_the_given_status() {
        let response = ApiError::new(StatusCode::CONFLICT, "nope").into_response();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/plain; charset=utf-8")
        );
    }

    #[test]
    fn clip_never_splits_a_character() {
        // 20 multi-byte characters: a naive &value[..16] panics here.
        let clipped = clip(&"あ".repeat(20), 16);
        assert!(clipped.ends_with("..."));
        assert!(clipped.len() <= 19);
    }

    #[test]
    fn clip_leaves_short_values_alone() {
        assert_eq!(clip("manga-ocr", 16), "manga-ocr");
    }

    #[test]
    fn clip_costs_three_bytes_more_than_its_limit() {
        // The ellipsis is appended *after* the limit is filled. Pinning this is
        // the point: every caller sizes its budget around it.
        assert_eq!(clip(&"x".repeat(4096), 16).len(), 19);
    }

    /// A stand-in for a `PipelineError` chain: `Display` shows one level, the
    /// cause is only reachable through `source`.
    #[derive(Debug)]
    struct Layer {
        message: &'static str,
        source: Option<Box<Layer>>,
    }

    impl std::fmt::Display for Layer {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.message)
        }
    }

    impl std::error::Error for Layer {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.source
                .as_deref()
                .map(|source| source as &(dyn std::error::Error + 'static))
        }
    }

    fn layers(messages: &[&'static str]) -> Layer {
        let mut built: Option<Box<Layer>> = None;
        for message in messages.iter().rev() {
            built = Some(Box::new(Layer {
                message,
                source: built,
            }));
        }
        *built.expect("a chain needs at least one layer")
    }

    #[test]
    fn a_two_level_source_chain_surfaces_its_innermost_message() {
        // Ollama down. Display stops at the stage context; "connection refused"
        // is two levels below it.
        let error = layers(&[
            "pipeline translation failed: qwen3:8b failed",
            "error sending request for url (http://localhost:11434/v1/chat/completions)",
            "tcp connect error: Connection refused (os error 10061)",
        ]);
        assert_eq!(
            root_cause(&error),
            "tcp connect error: Connection refused (os error 10061)"
        );
        // A parenthesised URL must not be mistaken for a Debug newtype.
        assert_eq!(
            root_cause(error.source().unwrap()),
            "tcp connect error: Connection refused (os error 10061)"
        );
    }

    #[test]
    fn the_root_cause_is_still_visible_after_clipping() {
        let error = layers(&[
            "pipeline ocr failed: failed to load paddleocr-vl-1.6",
            "No space left on device (os error 28)",
        ]);
        let message = clip(&root_cause(&error), CLIP_BUDGET);
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
        assert!(message.contains("No space left on device"), "{message}");
    }

    #[test]
    fn an_error_without_a_source_is_left_alone() {
        let error = layers(&["nothing below me"]);
        assert_eq!(root_cause(&error), "nothing below me");
    }

    #[test]
    fn the_log_keeps_the_links_the_wire_cannot_afford() {
        let error = layers(&["pipeline ocr failed: manga-ocr failed", "os error 28"]);
        assert_eq!(
            chain(&error),
            "pipeline ocr failed: manga-ocr failed <- os error 28"
        );
    }

    /// The failure this whole path exists for, written out exactly as it arrives:
    /// koharu's stage head, tokio's join wrapper, `Result::unwrap`'s prefix and
    /// koharu-torch's `Torch(...)` newtype, around the cuDNN text the user has to
    /// see. Built from `run.exe`'s own output.
    const CUDNN: &str = "CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed \
                         ptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH";

    #[test]
    fn the_real_cudnn_failure_names_cudnn_inside_the_extensions_window() {
        let error = layers(&[
            "pipeline detection failed: koharu-layout-rfdetr-seg-2xl failed",
            "layout detection task panicked",
            // `Box::leak` because `Layer` holds a `&'static str`; a test process
            // that ends is the only cleanup this needs.
            Box::leak(
                format!(
                    "task 12 panicked with message \"called `Result::unwrap()` on an `Err` value: \
                     Torch(\\\"{CUDNN}\\\")\""
                )
                .into_boxed_str(),
            ),
        ]);
        assert_eq!(root_cause(&error), CUDNN);

        // What the extension actually renders. The old message spent all 120
        // bytes on the head and the scaffolding and never reached the C in CUDNN.
        let message = clip(&format!("detection failed: {}", root_cause(&error)), CLIP_BUDGET);
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
        assert!(message.starts_with("detection failed: CUDNN_BACKEND"), "{message}");
        assert!(message.contains("cudnnFinalize"), "{message}");
    }

    #[tokio::test]
    async fn a_genuine_tokio_join_error_is_peeled_to_the_panic_itself() {
        // Not a hand-written string: tokio formats the join error and the
        // standard library formats the unwrap, so this pins the real wrappers
        // rather than our idea of them. Only the innermost value is ours, and it
        // is the shape koharu-torch produces.
        struct Torch(&'static str);

        // Written out rather than derived, so the shape being parsed is stated
        // here rather than assumed of the derive.
        impl std::fmt::Debug for Torch {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "Torch({:?})", self.0)
            }
        }

        let panicked = tokio::spawn(async { Err::<(), Torch>(Torch(CUDNN)).unwrap() })
            .await
            .expect_err("the task panicked");
        assert_eq!(strip_panic_scaffolding(&panicked.to_string()), CUDNN);
        assert_eq!(panic_text(&*panicked.into_panic()), CUDNN);
    }

    #[test]
    fn an_ordinary_message_is_left_exactly_as_it_is() {
        for message in [
            "tcp connect error: Connection refused (os error 10061)",
            "No space left on device (os error 28)",
            "task queue full",
            "",
        ] {
            assert_eq!(strip_panic_scaffolding(message), message, "{message}");
        }
    }

    #[test]
    fn a_payload_with_no_text_still_produces_a_readable_body() {
        // `panic_any` is legal and carries no message; the alternative was
        // printing `{:?}` of an unbounded value into a 120-byte window.
        let payload: Box<dyn Any + Send> = Box::new(42_u32);
        assert_eq!(panic_text(&*payload), UNPRINTABLE_PANIC);
    }

    #[test]
    fn both_string_payload_shapes_are_read() {
        // `panic!("literal")` stores a &str, `panic!("{x}")` a String.
        let borrowed: Box<dyn Any + Send> = Box::new("plain trouble");
        let owned: Box<dyn Any + Send> = Box::new("plain trouble".to_owned());
        assert_eq!(panic_text(&*borrowed), "plain trouble");
        assert_eq!(panic_text(&*owned), "plain trouble");
    }

    #[test]
    fn a_wholly_reflected_error_fits_the_window() {
        for value in ["x".repeat(4096), "あ".repeat(4096)] {
            let message = clip(&value, CLIP_BUDGET);
            assert!(message.len() <= BUDGET, "{} bytes", message.len());
        }
    }
}
