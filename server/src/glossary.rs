//! Pinned per-series term renderings, keyed by the same story id as the
//! context window.
//!
//! A story pair is *what an earlier page happened to say*, offered to the model
//! as precedent; a glossary entry is *what this term is called*, law for every
//! occurrence. The distinction was measured before it was built: the story
//! window propagates a good first rendering and cannot repair a bad one — one
//! rank term rendered four ways across four sites, the window unsaturated —
//! while an 8-term prototype passed as pipeline instructions delivered the
//! official term at every registered site over both test chapters.
//! Instructions are snapshotted config, though — changing them reloads the
//! pipeline — so the productised channel travels per request beside `context`
//! and is fed from this store.
//!
//! Deliberately a sibling of [`crate::story::Stories`], not a tenant of it: the
//! popup's "New story" clears the *context* a reading session accumulated, but
//! the terms a series uses are stable across sessions — the id is minted
//! deterministically from host and series, so the same series always finds its
//! glossary again. Like the story store this is in-memory only and dies with
//! the process; the caller that installed it can re-install it.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use koharu_translator::TranslationContext;

/// How many series keep a glossary at once. Mirrors `MAX_STORIES`, for the same
/// reason: a reader has one series open, and the bound exists so stale ids
/// cannot grow the map without limit.
const MAX_GLOSSARIES: usize = 8;

/// Most terms one glossary may hold. The measured prototype used 8; the bound
/// is generous because a long series legitimately accretes names, but it must
/// exist — every term rides in every page's prompt, and prompt tokens are
/// charged at the KV rate the story window's own doc comment records.
pub const MAX_TERMS: usize = 64;

/// Longest source or translation accepted for one term, in bytes. A term is a
/// name, not a paragraph; the bound keeps a hostile or confused caller from
/// storing a page of text that would then be serialized into every prompt.
pub const MAX_TERM_LEN: usize = 256;

/// The per-series glossaries. One `Mutex` over the whole map, held only for the
/// length of a clone or an insert — the same locking argument as `Stories`.
#[derive(Default)]
pub struct Glossaries {
    inner: Mutex<HashMap<String, Vec<TranslationContext>>>,
}

impl Glossaries {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The terms to pin for `id`, in installation order.
    ///
    /// An unknown id is not an error — most series have no glossary, and that
    /// must cost nothing.
    #[must_use]
    pub fn terms(&self, id: &str) -> Arc<[TranslationContext]> {
        let Ok(glossaries) = self.inner.lock() else {
            return Arc::from([] as [TranslationContext; 0]);
        };
        glossaries.get(id).map_or_else(
            || Arc::from([] as [TranslationContext; 0]),
            |terms| Arc::from(terms.as_slice()),
        )
    }

    /// Installs the glossary for `id`, replacing whatever was there. Empty
    /// clears — one verb, so the caller cannot half-update.
    ///
    /// Returns how many terms are now stored. The caller validates count and
    /// lengths ([`parse_terms`]); this only bounds the map itself.
    pub fn set(&self, id: &str, terms: Vec<TranslationContext>) -> usize {
        debug_assert!(terms.len() <= MAX_TERMS);
        let Ok(mut glossaries) = self.inner.lock() else {
            return 0;
        };
        if terms.is_empty() {
            glossaries.remove(id);
            return 0;
        }
        if !glossaries.contains_key(id) && glossaries.len() >= MAX_GLOSSARIES {
            // Whole glossaries, fewest terms first — the same shape as the
            // story store's eviction, for the same reason: it cannot truncate
            // the series being read right now.
            if let Some(victim) = glossaries
                .iter()
                .min_by_key(|(_, terms)| terms.len())
                .map(|(key, _)| key.clone())
            {
                glossaries.remove(&victim);
            }
        }
        let stored = terms.len();
        glossaries.insert(id.to_owned(), terms);
        stored
    }

    /// How many terms a series is carrying. For tests.
    #[must_use]
    pub fn len(&self, id: &str) -> usize {
        self.inner
            .lock()
            .map_or(0, |glossaries| glossaries.get(id).map_or(0, Vec::len))
    }
}

/// One term as `/glossary` receives it.
#[derive(serde::Deserialize)]
struct TermIn {
    source: String,
    translation: String,
}

/// Parses and validates the `terms` field of a `/glossary` request: a JSON
/// array of `{source, translation}`.
///
/// Rejection over truncation, everywhere — a glossary silently shortened would
/// pin some terms and not others with nothing on the wire saying so, which is
/// the exact silent-loss shape this server is designed never to produce.
pub fn parse_terms(json: &str) -> Result<Vec<TranslationContext>, String> {
    let terms: Vec<TermIn> = serde_json::from_str(json)
        .map_err(|error| format!("terms must be a JSON array of {{source, translation}}: {error}"))?;
    if terms.len() > MAX_TERMS {
        return Err(format!("at most {MAX_TERMS} terms per glossary, got {}", terms.len()));
    }
    let mut parsed = Vec::with_capacity(terms.len());
    for (index, term) in terms.into_iter().enumerate() {
        let source = term.source.trim();
        let translation = term.translation.trim();
        if source.is_empty() || translation.is_empty() {
            return Err(format!("term {index} has an empty source or translation"));
        }
        if source.len() > MAX_TERM_LEN || translation.len() > MAX_TERM_LEN {
            return Err(format!("term {index} exceeds {MAX_TERM_LEN} bytes"));
        }
        parsed.push(TranslationContext {
            source: source.to_owned(),
            translation: translation.to_owned(),
        });
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(source: &str, translation: &str) -> TranslationContext {
        TranslationContext {
            source: source.to_owned(),
            translation: translation.to_owned(),
        }
    }

    #[test]
    fn terms_come_back_in_installation_order() {
        let glossaries = Glossaries::default();
        let stored = glossaries.set(
            "series-1",
            vec![term("청풍검대", "the Clear Wind Swords"), term("흑월문", "the Black Moon Sect")],
        );
        assert_eq!(stored, 2);
        let terms = glossaries.terms("series-1");
        assert_eq!(terms.len(), 2);
        assert_eq!(terms[0].source, "청풍검대");
        assert_eq!(terms[0].translation, "the Clear Wind Swords");
        assert_eq!(terms[1].source, "흑월문");
    }

    #[test]
    fn an_unknown_series_carries_nothing() {
        let glossaries = Glossaries::default();
        assert!(glossaries.terms("never-seen").is_empty());
    }

    #[test]
    fn set_replaces_rather_than_appends() {
        let glossaries = Glossaries::default();
        glossaries.set("series-1", vec![term("a", "A"), term("b", "B")]);
        glossaries.set("series-1", vec![term("c", "C")]);
        let terms = glossaries.terms("series-1");
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].source, "c");
    }

    #[test]
    fn an_empty_set_clears_the_series() {
        let glossaries = Glossaries::default();
        glossaries.set("series-1", vec![term("a", "A")]);
        assert_eq!(glossaries.set("series-1", Vec::new()), 0);
        assert!(glossaries.terms("series-1").is_empty());
        assert_eq!(glossaries.len("series-1"), 0);
    }

    #[test]
    fn glossaries_are_bounded_so_stale_ids_cannot_grow_without_limit() {
        let glossaries = Glossaries::default();
        for index in 0..MAX_GLOSSARIES {
            // Two terms each, so the one-term latecomer below cannot be its
            // own eviction victim.
            glossaries.set(&format!("series-{index}"), vec![term("a", "A"), term("b", "B")]);
        }
        glossaries.set("one-more", vec![term("z", "Z")]);
        let held = (0..MAX_GLOSSARIES)
            .filter(|index| !glossaries.terms(&format!("series-{index}")).is_empty())
            .count();
        assert_eq!(held, MAX_GLOSSARIES - 1, "one resident should have been evicted");
        assert_eq!(glossaries.len("one-more"), 1, "the newcomer must be stored");
    }

    #[test]
    fn parse_accepts_the_measured_prototype_shape() {
        let parsed = parse_terms(
            r#"[{"source":"청풍검대","translation":"the Clear Wind Swords"},
                {"source":"유성검법","translation":"the Falling Star Sword Art"}]"#,
        )
        .unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].translation, "the Falling Star Sword Art");
    }

    #[test]
    fn parse_rejects_rather_than_truncates() {
        assert!(parse_terms("not json").is_err());
        assert!(parse_terms(r#"[{"source":"","translation":"x"}]"#).is_err());
        assert!(parse_terms(r#"[{"source":"x","translation":"  "}]"#).is_err());
        let over = format!(
            "[{}]",
            (0..=MAX_TERMS)
                .map(|i| format!(r#"{{"source":"s{i}","translation":"t{i}"}}"#))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(parse_terms(&over).is_err());
        let long = "x".repeat(MAX_TERM_LEN + 1);
        assert!(parse_terms(&format!(r#"[{{"source":"{long}","translation":"t"}}]"#)).is_err());
    }
}
