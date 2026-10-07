//! Origin, Host and token checks.
//!
//! Loopback is not isolation: any web page's JavaScript can `fetch`
//! http://127.0.0.1:8765, and a cross-origin POST carrying `FormData` is a CORS
//! simple request, so it reaches the handler even though the page cannot read
//! the reply. The guards run as middleware, before a request can reach anything
//! that touches a model.
//!
//! Every decision lives in a pure function so the policy is testable without a
//! server, a listener, or a pipeline.

use std::{net::SocketAddr, sync::Arc};

use axum::{
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::Response,
};

use crate::error::{ApiError, BAD_HOST, UNAUTHORIZED, clip};

/// Firefox's own extension scheme. The extension holds host permissions for
/// 127.0.0.1, so its background fetch is CORS-exempt and sends either this or
/// no Origin at all.
const EXTENSION_SCHEME: &str = "moz-extension://";

pub const BAD_ORIGIN_SUFFIX: &str =
    "not the BireLate extension; restart with --allow-origin to permit it";

/// How much of a rejected Origin is echoed. Enough to recognise the page,
/// short enough to keep the advice inside the extension's window.
const ECHO: usize = 20;

/// HTTP's default port, which a browser leaves out of the `Host` header.
const HTTP_DEFAULT_PORT: u16 = 80;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Verdict {
    Allow,
    Deny,
}

impl Verdict {
    const fn allowed(self) -> bool {
        matches!(self, Self::Allow)
    }
}

/// The slice of server state the guards need. Kept separate from `AppState` so
/// the policy can be exercised without constructing a `Pipeline`.
#[derive(Clone)]
pub struct GuardState {
    /// `None` means `--no-token`: authentication is off.
    pub token_digest: Option<[u8; 32]>,
    pub allowed_origins: Arc<[String]>,
    pub allowed_hosts: Arc<[String]>,
}

/// Rejects any Origin that is not the extension's own.
///
/// `Origin: null` is denied, and that is load-bearing. A page at
/// https://evil.example doing `fetch(url, {mode: "no-cors", referrerPolicy:
/// "no-referrer", body: formData})` sends `null` rather than its real origin,
/// because Fetch replaces the serialized origin for a non-`cors` request. An
/// *absent* Origin is safe to allow by contrast: the specification always
/// appends the header on a non-GET/HEAD request, so a page cannot omit it.
#[must_use]
pub fn origin_verdict(origin: Option<&str>, allowed: &[String]) -> Verdict {
    let Some(origin) = origin else {
        return Verdict::Allow;
    };
    if allowed.iter().any(|entry| entry == origin) || origin.starts_with(EXTENSION_SCHEME) {
        Verdict::Allow
    } else {
        Verdict::Deny
    }
}

/// Blunts DNS rebinding, where an attacker serves a page from
/// http://attacker.test:8765, repoints the name at 127.0.0.1, and thereby
/// becomes same-origin with us -- the one vector that could read output.
/// `origin_verdict` already refuses that request; this is the second layer.
#[must_use]
pub fn host_verdict(host: Option<&str>, allowed: &[String]) -> Verdict {
    match host {
        Some(host) if allowed.iter().any(|entry| entry.eq_ignore_ascii_case(host)) => Verdict::Allow,
        _ => Verdict::Deny,
    }
}

/// Compares the shared secret in constant time.
///
/// `blake3::Hash`'s equality is constant time, and hashing both sides also
/// removes the length leak a byte-wise comparison of the raw tokens would have.
#[must_use]
pub fn token_verdict(presented: Option<&[u8]>, expected: Option<[u8; 32]>) -> Verdict {
    let Some(expected) = expected else {
        return Verdict::Allow;
    };
    match presented {
        Some(presented) if blake3::hash(presented) == expected => Verdict::Allow,
        _ => Verdict::Deny,
    }
}

#[must_use]
pub fn digest(token: &str) -> [u8; 32] {
    *blake3::hash(token.as_bytes()).as_bytes()
}

/// The `Host` values a client may legitimately use to reach `addr`.
///
/// An unspecified bind -- `0.0.0.0` or `[::]` -- is deliberately not seeded with
/// its own spelling. Those are bind addresses, never destinations: no browser
/// ever sends `Host: 0.0.0.0:8765`, so a list containing only that matches
/// nothing and every request 403s, `/health` included, which the popup can only
/// report as "no server here". Such a bind does listen on loopback, so the
/// loopback spellings are the ones that belong in the list. A LAN address still
/// needs `--allow-host`, because the interfaces cannot be enumerated from a
/// `SocketAddr`.
#[must_use]
pub fn allowed_hosts_from(addr: SocketAddr) -> Vec<String> {
    let mut hosts = Vec::new();
    let mut push = |host: String| {
        if !hosts.contains(&host) {
            hosts.push(host);
        }
    };
    if !addr.ip().is_unspecified() {
        push(addr.to_string());
    }
    if addr.ip().is_loopback() || addr.ip().is_unspecified() {
        push(format!("127.0.0.1:{}", addr.port()));
        push(format!("localhost:{}", addr.port()));
        push(format!("[::1]:{}", addr.port()));
    }
    // A browser omits the scheme's default port from `Host`, so binding :80
    // would otherwise match nothing and 403 every request, `/health` included --
    // which the popup can only report as "no server here".
    if addr.port() == HTTP_DEFAULT_PORT {
        if !addr.ip().is_unspecified() {
            push(bare_host(addr));
        }
        if addr.ip().is_loopback() || addr.ip().is_unspecified() {
            push("127.0.0.1".to_owned());
            push("localhost".to_owned());
            push("[::1]".to_owned());
        }
    }
    hosts
}

/// The host part of `addr` on its own, keeping IPv6's brackets: `Host` uses the
/// same bracketed form the authority does.
fn bare_host(addr: SocketAddr) -> String {
    match addr {
        SocketAddr::V4(addr) => addr.ip().to_string(),
        SocketAddr::V6(addr) => format!("[{}]", addr.ip()),
    }
}

pub async fn guard_origin_host(
    State(guard): State<GuardState>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let headers = request.headers();
    let origin = headers
        .get(header::ORIGIN)
        .map(|value| value.to_str().unwrap_or("<not utf-8>"));
    let host = headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok());
    let text = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("<absent>")
    };

    // Logged on every request because what Firefox attaches to an extension's
    // own fetch is not documented: one real translate attempt settles it, and a
    // rejected request never reaches a model. The Sec-Fetch-* pair is recorded
    // alongside it as diagnostics only -- never branched on, because the
    // extension's own fetch is cross-site relative to 127.0.0.1 and so carries
    // exactly the values a hostile page would.
    tracing::info!(
        path = %request.uri().path(),
        origin = origin.unwrap_or("<absent>"),
        host = host.unwrap_or("<absent>"),
        sec_fetch_site = text("sec-fetch-site"),
        sec_fetch_mode = text("sec-fetch-mode"),
        "request"
    );

    if !origin_verdict(origin, &guard.allowed_origins).allowed() {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            format!(
                "rejected Origin \"{}\": {BAD_ORIGIN_SUFFIX}",
                clip(origin.unwrap_or_default(), ECHO)
            ),
        ));
    }
    if !host_verdict(host, &guard.allowed_hosts).allowed() {
        return Err(ApiError::new(StatusCode::FORBIDDEN, BAD_HOST));
    }
    Ok(next.run(request).await)
}

pub async fn require_token(
    State(guard): State<GuardState>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let presented = request
        .headers()
        .get("x-koharu-token")
        .map(axum::http::HeaderValue::as_bytes);
    if !token_verdict(presented, guard.token_digest).allowed() {
        // No WWW-Authenticate header: it would make Firefox raise a native
        // credential dialog on a navigation to this port.
        return Err(ApiError::new(StatusCode::UNAUTHORIZED, UNAUTHORIZED));
    }
    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none() -> Vec<String> {
        Vec::new()
    }

    #[test]
    fn an_absent_origin_is_the_extension() {
        assert_eq!(origin_verdict(None, &none()), Verdict::Allow);
    }

    #[test]
    fn the_extension_scheme_is_allowed() {
        assert_eq!(
            origin_verdict(
                Some("moz-extension://2b4c1f6e-0000-4000-8000-abcdefabcdef"),
                &none()
            ),
            Verdict::Allow
        );
    }

    #[test]
    fn a_web_page_is_rejected() {
        assert_eq!(
            origin_verdict(Some("https://evil.example"), &none()),
            Verdict::Deny
        );
    }

    #[test]
    fn a_null_origin_is_rejected() {
        // A no-cors, no-referrer POST from any https page arrives as "null",
        // so allowing it would be a complete bypass of this guard.
        assert_eq!(origin_verdict(Some("null"), &none()), Verdict::Deny);
    }

    #[test]
    fn a_page_served_from_our_own_port_is_still_a_page() {
        assert_eq!(
            origin_verdict(Some("http://127.0.0.1:8765"), &none()),
            Verdict::Deny
        );
    }

    #[test]
    fn the_extension_check_is_on_the_scheme_not_a_substring() {
        assert_eq!(
            origin_verdict(Some("https://moz-extension.example"), &none()),
            Verdict::Deny
        );
    }

    #[test]
    fn an_explicitly_allowed_origin_matches_exactly() {
        let allowed = vec!["null".to_owned()];
        assert_eq!(origin_verdict(Some("null"), &allowed), Verdict::Allow);
        assert_eq!(origin_verdict(Some("NULL"), &allowed), Verdict::Deny);
    }

    #[test]
    fn loopback_hosts_cover_every_spelling_of_the_bind_address() {
        assert_eq!(
            allowed_hosts_from("127.0.0.1:8765".parse().unwrap()),
            vec![
                "127.0.0.1:8765".to_owned(),
                "localhost:8765".to_owned(),
                "[::1]:8765".to_owned(),
            ]
        );
    }

    #[test]
    fn an_unspecified_bind_matches_the_hosts_a_browser_actually_sends() {
        // `--addr 0.0.0.0:8765` is permitted whenever a token is set, and the
        // extension still points at http://127.0.0.1:8765. Seeding the list
        // with "0.0.0.0:8765" and nothing else made every request a 403.
        let allowed = allowed_hosts_from("0.0.0.0:8765".parse().unwrap());
        assert!(
            !allowed.iter().any(|host| host.starts_with("0.0.0.0")),
            "{allowed:?}"
        );
        assert_eq!(
            host_verdict(Some("127.0.0.1:8765"), &allowed),
            Verdict::Allow
        );
        assert_eq!(host_verdict(Some("localhost:8765"), &allowed), Verdict::Allow);
        assert_eq!(host_verdict(Some("evil.test:8765"), &allowed), Verdict::Deny);
    }

    #[test]
    fn an_unspecified_ipv6_bind_covers_loopback_as_well() {
        let allowed = allowed_hosts_from("[::]:8765".parse().unwrap());
        assert!(!allowed.iter().any(|host| host.starts_with("[::]")), "{allowed:?}");
        assert_eq!(host_verdict(Some("[::1]:8765"), &allowed), Verdict::Allow);
        assert_eq!(
            host_verdict(Some("127.0.0.1:8765"), &allowed),
            Verdict::Allow
        );
    }

    #[test]
    fn binding_the_default_port_still_matches_what_a_browser_sends() {
        // Firefox sends `Host: 127.0.0.1` for http://127.0.0.1/, with no port.
        let allowed = allowed_hosts_from("127.0.0.1:80".parse().unwrap());
        assert_eq!(host_verdict(Some("127.0.0.1"), &allowed), Verdict::Allow);
        assert_eq!(host_verdict(Some("localhost"), &allowed), Verdict::Allow);
        assert_eq!(host_verdict(Some("127.0.0.1:80"), &allowed), Verdict::Allow);
        assert_eq!(host_verdict(Some("evil.test"), &allowed), Verdict::Deny);
        // Any other port keeps the strict spelling.
        let allowed = allowed_hosts_from("127.0.0.1:8765".parse().unwrap());
        assert_eq!(host_verdict(Some("127.0.0.1"), &allowed), Verdict::Deny);
    }

    #[test]
    fn a_specific_non_loopback_bind_keeps_only_its_own_address() {
        let allowed = allowed_hosts_from("192.0.2.5:8765".parse().unwrap());
        assert_eq!(
            host_verdict(Some("192.0.2.5:8765"), &allowed),
            Verdict::Allow
        );
        assert_eq!(host_verdict(Some("127.0.0.1:8765"), &allowed), Verdict::Deny);
    }

    #[test]
    fn host_matching_is_case_insensitive_and_requires_the_header() {
        let allowed = allowed_hosts_from("127.0.0.1:8765".parse().unwrap());
        assert_eq!(host_verdict(Some("LOCALHOST:8765"), &allowed), Verdict::Allow);
        assert_eq!(host_verdict(Some("evil.test:8765"), &allowed), Verdict::Deny);
        assert_eq!(host_verdict(None, &allowed), Verdict::Deny);
    }

    #[test]
    fn the_token_gates_only_when_one_is_configured() {
        let expected = digest("s3cret");
        assert_eq!(
            token_verdict(Some(b"s3cret"), Some(expected)),
            Verdict::Allow
        );
        assert_eq!(token_verdict(Some(b"wrong"), Some(expected)), Verdict::Deny);
        assert_eq!(token_verdict(None, Some(expected)), Verdict::Deny);
        assert_eq!(token_verdict(None, None), Verdict::Allow);
        assert_eq!(token_verdict(Some(b"anything"), None), Verdict::Allow);
    }

    #[test]
    fn token_comparison_handles_any_presented_length() {
        let expected = digest("s3cret");
        assert_eq!(token_verdict(Some(b"x"), Some(expected)), Verdict::Deny);
        let long = vec![b'x'; 4096];
        assert_eq!(token_verdict(Some(&long), Some(expected)), Verdict::Deny);
    }
}
