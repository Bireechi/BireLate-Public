//! Turning one uploaded image into a one-page scene.
//!
//! This mirrors `koharu-pipeline`'s own `run` binary: a fresh in-memory session
//! per request is what makes the handler stateless, and the pipeline only ever
//! sees a page whose dimensions came straight from the decoded upload.

use std::{collections::BTreeMap, sync::Arc};

use anyhow::Result;
use koharu_pipeline::{Committer, StageOutput};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, EntityId, PageDraft, Session, Snapshot,
};

use crate::error::{ApiError, UNSUPPORTED_IMAGE};

/// The renderer's surface limits. They live in private modules of
/// koharu-renderer, where exceeding them is a hard error rather than a resize,
/// so they are checked here to produce a message the extension can show.
///
/// Two gates apply and the looser one is not the binding one.
/// `compositor::surface_size` refuses a side over `MAX_SURFACE_DIMENSION` or a
/// total over `MAX_SURFACE_PIXELS`. `Rasterizer::readback` then refuses a side
/// over the wgpu device's `max_texture_dimension_2d`, and vello builds that
/// device with `wgpu::Limits::default()`, which pins the value at 8192 however
/// capable the card is. 8192 is the smaller cap on every axis -- 8192*8192 is
/// well under the pixel total -- so it is the limit that actually fires, and it
/// fires *last*, after detection, OCR, the translation LLM and inpainting have
/// all been paid for. An 800x12000 webtoon strip is ordinary input; checking it
/// here turns several GPU-minutes ending in a 500 into an instant 400.
const MAX_SURFACE_DIMENSION: u32 = 32_768;
const MAX_SURFACE_PIXELS: u64 = 268_435_456;
const MAX_TEXTURE_DIMENSION: u32 = 8_192;

pub struct PreparedPage {
    pub session: Session,
    pub page: EntityId,
}

/// Deletes `koharu-storage-*` directories left behind by earlier runs.
///
/// **`Session::memory()` is not in memory.** koharu-storage opens RocksDB in a
/// `tempfile::TempDir` named with that prefix (`koharu-storage/src/database.rs`),
/// and the uploaded page goes in as an *uncompressed* blob -- so every request
/// writes the reader's manga to `%TEMP%` for the life of the request. The
/// directory is removed when the `Engine` drops, which is correct for every
/// ordinary exit; it leaks when the process dies without unwinding, and this
/// project has a documented mode for exactly that (the native `abort()` that
/// exits 3 with no output). A leaked directory holds a complete copy of the
/// page it was serving.
///
/// Called once at startup rather than per request, and it never touches a
/// directory whose lock file is held, so a second BireLate server on the same
/// machine is not sabotaged.
pub fn sweep_stale_storage() -> usize {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let is_ours = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("koharu-storage-"));
        if !is_ours {
            continue;
        }
        /* A live server holds LOCK open, and Windows refuses to remove a
         * directory with an open handle in it -- so the failure is the guard.
         * Nothing here decides whether another process is running; it just
         * declines to delete what it cannot. */
        if std::fs::remove_dir_all(&path).is_ok() {
            removed += 1;
            tracing::info!(path = %path.display(), "removed a stale koharu-storage directory");
        }
    }
    removed
}

/// Decodes the upload and commits it as the page's `source` asset.
///
/// Blocking: image decoding and the in-memory storage engine both are. Call it
/// from `spawn_blocking`, outside the GPU gate, so decode cost never queues
/// behind someone else's translation.
pub fn prepare_page(bytes: &[u8]) -> Result<PreparedPage, ApiError> {
    let format = image::guess_format(bytes).map_err(|_| ApiError::bad_request(UNSUPPORTED_IMAGE))?;
    let decoded = image::load_from_memory_with_format(bytes, format)
        .map_err(|_| ApiError::bad_request(UNSUPPORTED_IMAGE))?;
    let (width, height) = (decoded.width(), decoded.height());
    check_surface(width, height)?;

    let mut session = Session::memory().map_err(scene_failed)?;
    let mut page = None;
    let patch = session
        .snapshot()
        .patch(|edit| {
            let id = edit.add_page(
                PageDraft::new("page", f64::from(width), f64::from(height)),
                At::End,
            )?;
            edit.set_asset(
                id,
                &AssetRole::new("source")?,
                AssetInput::new(
                    Arc::<[u8]>::from(bytes),
                    format.to_mime_type(),
                    AssetMetadata {
                        width: Some(width),
                        height: Some(height),
                        attributes: BTreeMap::new(),
                    },
                ),
            )?;
            page = Some(id);
            Ok(())
        })
        .map_err(scene_failed)?;
    session.commit(patch).map_err(scene_failed)?;

    Ok(PreparedPage {
        session,
        // The closure above ran to completion, so the edit assigned an id.
        page: page.ok_or_else(|| ApiError::internal("the page was not added to the scene"))?,
    })
}

fn check_surface(width: u32, height: u32) -> Result<(), ApiError> {
    // Whichever of the two gates is tighter, so the check follows if either
    // constant is ever revised.
    let cap = MAX_SURFACE_DIMENSION.min(MAX_TEXTURE_DIMENSION);
    if width == 0
        || height == 0
        || width > cap
        || height > cap
        || u64::from(width) * u64::from(height) > MAX_SURFACE_PIXELS
    {
        return Err(ApiError::bad_request(format!(
            "page is {width}x{height}; the renderer caps a side at {cap} pixels \
             (split long-strip pages before uploading)"
        )));
    }
    Ok(())
}

fn scene_failed(error: koharu_scene::Error) -> ApiError {
    ApiError::internal(crate::error::clip(
        &format!("could not build the page: {error}"),
        crate::error::CLIP_BUDGET,
    ))
}

/// Folds every stage patch back into the session, so the snapshot the renderer
/// and the region reader see is the finished one.
pub struct SessionCommitter<'a>(pub &'a mut Session);

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
        Ok(self.0.commit(output.patch)?.snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::BUDGET;

    #[test]
    fn an_oversized_page_is_refused_with_a_readable_message() {
        let message = check_surface(40_000, 100).unwrap_err().message;
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
        assert!(message.contains("40000x100"));
    }

    #[test]
    fn a_page_within_the_limits_is_accepted() {
        assert!(check_surface(1_600, 2_300).is_ok());
    }

    #[test]
    fn a_long_strip_page_is_refused_before_the_gpu_is_touched() {
        // Ordinary webtoon shape. It clears the compositor's 32768-per-side cap
        // and its pixel total, so only the rasterizer's 8192 stops it -- and
        // that runs after the whole translation has already been computed.
        let message = check_surface(800, 12_000).unwrap_err().message;
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
        assert!(message.contains("8192"), "{message}");
        assert!(message.contains("800x12000"), "{message}");
    }

    #[test]
    fn the_largest_representable_page_still_fits_the_extensions_window() {
        let message = check_surface(u32::MAX, u32::MAX).unwrap_err().message;
        assert!(message.len() <= BUDGET, "{} bytes: {message}", message.len());
    }

    #[test]
    fn undecodable_bytes_are_a_bad_request() {
        // `PreparedPage` owns a storage session, which is not worth making
        // printable just so `unwrap_err` can exist.
        let Err(error) = prepare_page(b"not an image at all") else {
            panic!("undecodable bytes should be rejected");
        };
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(error.message, UNSUPPORTED_IMAGE);
    }
}
