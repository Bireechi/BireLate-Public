//! Pinned English for drawn sound effects, applied to the finished scene.
//!
//! Same shape and the same reasoning as `lettering.rs`: it runs between
//! `Pipeline::execute` and the render, touches no device, and is a scene patch
//! rather than a Koharu change, so it costs no upstream patch.
//!
//! **What this is for, and what it is not.** Measured over 40 pages / 74
//! effects, cross-page *drift* on effects is essentially absent -- the same
//! source got the same English on 6 of 7 repeated strings. This is therefore
//! not a consistency fix. The defect it targets is that **8 of the 37 genuine
//! sound effects (22%) came back as romaji** -- `きゃっ` -> `Kya!`,
//! `ポィ` -> `Poi.` -- leaving the reader Japanese in Latin letters. That the
//! same source produced `Eek!` and `Kya!` inside one call is the evidence it is
//! the model being unreliable rather than the effect being untranslatable.
//!
//! **Restricted to onomatopoeia regions, and that gate is load-bearing.** The
//! imported dataset contains entries like `あっ` which are perfectly ordinary
//! dialogue; without the gate a two-kana entry would silently rewrite a spoken
//! line. The label is the detector's own, read back off the region the text was
//! recognised from.

use std::collections::HashMap;

use koharu_scene::{EntityId, Region, Session, TextLayout, Translation};

/// The detector's class for a drawn sound effect. Matches `regions.rs`.
const ONOMATOPOEIA: &str = "onomatopoeia";

/// Source-to-English pins, first file loaded winning.
#[derive(Debug, Default, Clone)]
pub struct Dictionary {
    entries: HashMap<String, String>,
}

#[derive(serde::Deserialize)]
struct DictionaryFile {
    entries: Vec<Entry>,
}

#[derive(serde::Deserialize)]
struct Entry {
    source: String,
    target: String,
}

impl Dictionary {
    /// Merges a file in. **Earlier files win**, so a small curated list can be
    /// passed before a large imported one and keep its overrides: the imported
    /// set is a coverage net, not an authority.
    pub fn merge_json(&mut self, json: &str) -> anyhow::Result<usize> {
        let parsed: DictionaryFile = serde_json::from_str(json)?;
        let before = self.entries.len();
        for entry in parsed.entries {
            let source = entry.source.trim();
            let target = entry.target.trim();
            if source.is_empty() || target.is_empty() {
                continue;
            }
            self.entries
                .entry(source.to_owned())
                .or_insert_with(|| target.to_owned());
        }
        Ok(self.entries.len() - before)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The pinned English for this source, if any.
    ///
    /// Trailing Japanese ellipses and the ASCII run OCR sometimes returns for
    /// them are stripped before lookup, because `ホ．．．` and `ホ` are the same
    /// effect and no dictionary should have to carry both.
    #[must_use]
    pub fn get(&self, source: &str) -> Option<&str> {
        let trimmed = source.trim();
        if let Some(hit) = self.entries.get(trimmed) {
            return Some(hit.as_str());
        }
        let stripped = trimmed
            .trim_end_matches(['．', '.', '。', '…', '・', '！', '!', '？', '?'])
            .trim();
        (stripped != trimmed)
            .then(|| self.entries.get(stripped).map(String::as_str))
            .flatten()
    }
}

/// Replaces the translation of every sound-effect region whose source is pinned.
///
/// Best-effort, exactly like `uppercase_dialogue`: a page whose scene cannot be
/// patched still renders, with the model's own wording. Returns how many layers
/// were rewritten, for the log.
pub fn pin_sound_effects(session: &mut Session, page: EntityId, dictionary: &Dictionary) -> usize {
    if dictionary.is_empty() {
        return 0;
    }
    let snapshot = session.snapshot();
    let Ok(descendants) = snapshot.descendants(page) else {
        return 0;
    };

    // Gathered by value before the patch opens, for the aliasing reason set out
    // in `lettering.rs`.
    let mut pending: Vec<(EntityId, Translation)> = Vec::new();
    for entity in descendants {
        if !matches!(entity.component::<TextLayout>(), Ok(Some(_))) {
            continue;
        }
        let Ok(layer) = snapshot.text_layer(entity.id()) else {
            continue;
        };
        let Ok(content) = layer.content() else { continue };

        // The gate. `source_region` is the region the text was recognised from,
        // and its `Region.label` is the detector's own class.
        let Ok(Some(region)) = content.source_region() else {
            continue;
        };
        let is_effect = snapshot
            .component::<Region>(region.id())
            .ok()
            .flatten()
            .is_some_and(|value| value.label.as_deref() == Some(ONOMATOPOEIA));
        if !is_effect {
            continue;
        }

        let Ok(Some(source)) = content.source() else {
            continue;
        };
        let Some(pinned) = dictionary.get(&source.text.value) else {
            continue;
        };
        let Ok(Some(translation)) = content.translation() else {
            continue;
        };
        if translation.text.value == pinned {
            continue;
        }

        let mut next = translation.clone();
        next.text.value = pinned.to_owned();
        pending.push((content.id(), next));
    }

    if pending.is_empty() {
        return 0;
    }
    let count = pending.len();
    let patched = snapshot.patch(|edit| {
        for (content, translation) in &pending {
            edit.set(*content, translation)?;
        }
        Ok(())
    });
    let Ok(patch) = patched else { return 0 };
    if session.commit(patch).is_ok() { count } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CURATED: &str = r#"{"entries":[{"source":"ピッ","target":"*Tap*"}]}"#;
    const IMPORTED: &str = r#"{"entries":[
        {"source":"ピッ","target":"beep"},
        {"source":"ドン","target":"thud"},
        {"source":"  ","target":"blank"},
        {"source":"ペラ","target":""}
    ]}"#;

    #[test]
    fn the_curated_file_wins_over_the_imported_one() {
        let mut dictionary = Dictionary::default();
        dictionary.merge_json(CURATED).unwrap();
        dictionary.merge_json(IMPORTED).unwrap();
        // The imported set is a coverage net, never an authority: it may only
        // fill gaps, or importing 3k entries would silently restyle the effects
        // the model already letters well.
        assert_eq!(dictionary.get("ピッ"), Some("*Tap*"));
        assert_eq!(dictionary.get("ドン"), Some("thud"));
    }

    #[test]
    fn blank_sources_and_targets_are_refused() {
        let mut dictionary = Dictionary::default();
        dictionary.merge_json(IMPORTED).unwrap();
        assert_eq!(dictionary.get("  "), None);
        assert_eq!(dictionary.get("ペラ"), None);
    }

    /// `ホ．．．` and `ホ` are one effect, and OCR returns both spellings.
    #[test]
    fn trailing_ellipses_and_marks_fall_back_to_the_bare_effect() {
        let mut dictionary = Dictionary::default();
        dictionary.merge_json(r#"{"entries":[{"source":"ホ","target":"*Phew*"}]}"#).unwrap();
        assert_eq!(dictionary.get("ホ．．．"), Some("*Phew*"));
        assert_eq!(dictionary.get("ホ..."), Some("*Phew*"));
        assert_eq!(dictionary.get(" ホ！！ "), Some("*Phew*"));
        // But it must not strip its way onto a different effect.
        assert_eq!(dictionary.get("ホホ"), None);
    }

    #[test]
    fn an_empty_dictionary_is_inert() {
        let dictionary = Dictionary::default();
        assert!(dictionary.is_empty());
        assert_eq!(dictionary.get("ピッ"), None);
    }
}
