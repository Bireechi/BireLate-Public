//! The edit apply's wording pins, applied to the finished scene.
//!
//! `lettering.rs`'s seam and shape: between `Pipeline::execute` and the
//! render, on the session the handler still owns -- gather by value from a
//! snapshot, one patch, one commit, best-effort. Runs BEFORE the scene gates
//! (`finish_scene` owns the order), so every gate judges the wording that
//! actually ships.
//!
//! WHY THIS EXISTS. An edit apply re-runs the whole page, and under the fixed
//! sampler seed any change to the request re-rolls every draw on it -- so
//! nudging one box reworded bubbles the reader never touched, and their
//! mental model ("I edited one box") did
//! not survive contact with the wire. The pin is the remedy: the extension
//! sends the entry's stored (source -> translated) pairs, and any region
//! whose OCR source comes back BYTE-IDENTICAL keeps its stored wording. A
//! region the edit changed reads differently, matches no pin, and translates
//! fresh -- which is the half the reader actually asked to change.
//!
//! Matching is byte-identity on the source string, nothing looser: a fuzzy
//! match would pin a genuinely different read to a stale translation, which
//! is worse than a re-roll. Two regions sharing one source string both pin
//! to the same wording -- deterministic, and what the model itself does.

use koharu_scene::{EntityId, Session, TextLayout, Translation};

/// Rewrites each matching region's translation to its pinned wording, and
/// returns how many regions were rewritten. A pin that matches nothing
/// counts nothing -- visible as asked != pinned in the log line.
pub fn apply_translation_pins(
    session: &mut Session,
    page: EntityId,
    pins: &[crate::routes::TranslationPin],
) -> usize {
    if pins.is_empty() {
        return 0;
    }
    let snapshot = session.snapshot();
    let Ok(descendants) = snapshot.descendants(page) else {
        return 0;
    };

    /* Gathered BEFORE the patch is opened, by value -- `lettering.rs`'s
     * aliasing rule. A text layer is the entity carrying a TextLayout. */
    let mut pending: Vec<(EntityId, Translation)> = Vec::new();
    for entity in descendants {
        if !matches!(entity.component::<TextLayout>(), Ok(Some(_))) {
            continue;
        }
        let Ok(layer) = snapshot.text_layer(entity.id()) else {
            continue;
        };
        let Ok(content) = layer.content() else { continue };
        let Ok(Some(source)) = content.source() else {
            continue;
        };
        let Some(pin) = pins.iter().find(|pin| pin.source == source.text.value) else {
            continue;
        };
        let Ok(Some(translation)) = content.translation() else {
            // A pin never invents a translation where the model produced
            // none: an absent component stays absent, and the drop report
            // stays true.
            continue;
        };
        if translation.text.value == pin.translation {
            continue; // the draw landed on the pinned wording by itself
        }
        let mut next = translation.clone();
        next.text.value = pin.translation.clone();
        pending.push((content.id(), next));
    }

    let pinned = pending.len();
    tracing::info!(
        asked = pins.len(),
        pinned,
        page = %page,
        "applying the edit apply's wording pins"
    );
    if pinned == 0 {
        return 0;
    }

    /* No generation, so `Edit::prepare_value` stamps `Origin::User` and the
     * write may overwrite the translation the pipeline authored -- the same
     * mechanism `lettering.rs:77-80` documents. */
    let patched = snapshot.patch(|edit| {
        for (content, translation) in &pending {
            edit.set(*content, translation)?;
        }
        Ok(())
    });
    let Ok(patch) = patched else { return 0 };
    if session.commit(patch).is_ok() { pinned } else { 0 }
}

#[cfg(test)]
mod tests {
    // The pass is exercised through `routes.rs`'s scene fixtures
    // (`page_with_layers`), beside the placement tests -- the observable is
    // `regions()`'s own `translated`, never `edit.set`'s `Ok`.
}
