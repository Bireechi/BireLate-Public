use std::{
    collections::BTreeMap,
    io::Cursor,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use image::{
    DynamicImage, GenericImageView as _, GrayImage, ImageFormat, Luma, Rgb, RgbImage, Rgba,
    RgbaImage,
};
use koharu_ml::{
    aot_inpainting::AotInpainting,
    flux2_klein::{Flux2KleinInpaint, Flux2KleinInpaintOptions},
    lama::{InpaintRequest, LaMa},
    rorem_mixed::{DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, RoremMixed, RoremMixedOptions},
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, BubbleRegion, EntityOrigin, Geometry, Origin,
    RasterLayer, RasterLayerKind, Region, RegionSpec, TextRegion,
};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::{ModelRef, StageInput, StageProcessor, finish, generation};
use crate::{InpaintingModel, ModelCell};

const PRODUCER: &str = "dev.koharu.pipeline.inpainting";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default, deny_unknown_fields)]
pub struct Flux2KleinConfig {
    pub prompt: String,

    /// How completely the masked region is replaced, 0..1.
    ///
    /// **Above 0.75 this field does nothing at all, and that is arithmetic
    /// rather than an observation.** `flux2_klein/mod.rs` converts it to a step
    /// boundary as
    /// `steps - floor(steps * (1 - strength))`, so at the distilled
    /// `num_inference_steps: 4` every value in `(0.75, 1.0]` floors to zero
    /// skipped steps and yields the identical boundary. 0.8, 0.999 and 1.0 are
    /// one setting. The axis only moves at <= 0.75 (3 steps), <= 0.5 (2) and
    /// <= 0.25 (1) -- i.e. toward *less* erasure, which is the wrong direction.
    ///
    /// **Verified rather than deduced**: the same seven pages at 0.8 and at
    /// 0.999 came out byte-identical, all seven PNGs.
    ///
    /// This corrects an earlier note here which claimed 0.999 measured *worse*
    /// (residue 105.7 against 93.1). That difference was **seed noise**: the
    /// 93.1 run predates the fixed `seed` below and was entropy-seeded. Two
    /// fixed-seed runs both give 105.7.
    ///
    /// Keep the number that follows, because it sets the floor for every future
    /// comparison: FLUX's run-to-run spread on one seed change was **12.6
    /// residue points, larger than the entire LaMa-to-RORem gap of 8.6**. An
    /// unseeded FLUX A/B can therefore invent an inpainter-sized effect out of
    /// nothing.
    pub strength: f64,

    /// Denoising steps. FLUX.2 klein is genuinely distilled, so 4 is its own
    /// default and is not the RORem situation.
    pub num_inference_steps: usize,

    /// Fixed, against a model default of `-1`. See `RoremMixedConfig::seed`.
    pub seed: i64,
}

impl Default for Flux2KleinConfig {
    fn default() -> Self {
        Self {
            prompt: "Remove the text and reconstruct the background.".to_owned(),
            strength: 0.8,
            num_inference_steps: 4,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default, deny_unknown_fields)]
pub struct RoremMixedConfig {
    pub prompt: String,
    pub negative_prompt: String,

    /// Square inference resolution; RORem accepts only 512 or 1024.
    ///
    /// **1024, against the model's own default of 512, and the difference is
    /// not a preference.** `Processor::resize_image` forces the tile to
    /// `resolution x resolution`, so the tiler's crops -- `TILE_SIZE` 512 plus
    /// `TILE_CONTEXT` 128 on each side, i.e. up to 768 -- are *downscaled* 1.5x
    /// at 512 and then stretched back afterwards. `inpaint_tiled` Lanczos-resizes
    /// any mismatch without complaining, so that never surfaces as an error; it
    /// just quietly costs detail. An A/B against LaMa at 512 measures the
    /// resampling, not the inpainter.
    ///
    /// The non-square squash is inherent to RORem and cannot be dialled out: a
    /// 768x300 crop is still made square. 1024 only stops the *scale* half of
    /// that being a handicap too.
    pub resolution: u32,

    /// Denoising steps per tile.
    ///
    /// RORem's own default is **30**, and that is the single reason it costs
    /// 35s a page where LaMa costs 0.6s -- the steps are paid per *tile*, and a
    /// dense page is several tiles. RORem is a *distilled* removal model, so the
    /// step count it actually needs is far below the SDXL convention its options
    /// inherit. Measured rather than assumed.
    pub num_inference_steps: i32,

    /// Sampling seed. Fixed, where the model's own default is `-1`.
    ///
    /// `-1` seeds from entropy, which would make two runs of the same page
    /// differ and put a random floor under every comparison. The A/A floor on
    /// this project is otherwise zero -- the translation seed is a constant
    /// upstream -- and that is what makes a single-run A/B readable at all.
    pub seed: i64,
}

impl Default for RoremMixedConfig {
    fn default() -> Self {
        Self {
            prompt: DEFAULT_PROMPT.to_owned(),
            negative_prompt: DEFAULT_NEGATIVE_PROMPT.to_owned(),
            resolution: 1024,
            num_inference_steps: 30,
            seed: 0,
        }
    }
}

pub(super) struct Processor {
    config: InpaintingModel,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
    /// Whether a glyph cut by the tile grid reaches the model whole.
    /// See `crop_tile_mask`.
    seam_safe: bool,
    /// `Some(dir)` writes the assembled inpaint masks there as PNGs.
    /// See `ProcessorConfig::debug_mask_dir`. `None` is the shipped state.
    debug_mask_dir: Option<PathBuf>,
}

impl Processor {
    pub(super) fn new(
        config: InpaintingModel,
        device: koharu_ml::Device,
        seam_safe: bool,
        debug_mask_dir: Option<String>,
    ) -> Result<Self> {
        match &config {
            InpaintingModel::LaMa {} | InpaintingModel::AotInpainting {} => {}
            InpaintingModel::Flux2Klein(settings) => {
                ensure!(
                    !settings.prompt.contains('\0'),
                    "FLUX.2 prompt contains NUL"
                );
            }
            InpaintingModel::RoremMixed(settings) => {
                ensure!(
                    !settings.prompt.contains('\0') && !settings.negative_prompt.contains('\0'),
                    "RORem prompt contains NUL"
                );
            }
        }

        Ok(Self {
            config,
            device,
            model: ModelCell::new(),
            seam_safe,
            debug_mask_dir: debug_mask_dir.map(PathBuf::from),
        })
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> ModelRef<'_> {
        let name = match self.config {
            InpaintingModel::LaMa {} => "lama",
            InpaintingModel::AotInpainting {} => "aot-inpainting",
            InpaintingModel::Flux2Klein(_) => "flux2-klein",
            InpaintingModel::RoremMixed(_) => "rorem-mixed",
        };
        ModelRef::new(name, &self.model)
    }

    async fn load(&self) -> Result<()> {
        self.model
            .ensure(|| Model::load(self.device.clone(), &self.config))
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("inpainting model is not loaded"))?
            .run(input, self.seam_safe, self.debug_mask_dir.as_deref())
            .await
    }
}

enum Model {
    LaMa(Arc<Mutex<LaMa>>),
    Aot(Arc<Mutex<AotInpainting>>),
    Flux {
        model: Arc<Mutex<Flux2KleinInpaint>>,
        config: Flux2KleinConfig,
    },
    Rorem {
        model: Arc<Mutex<RoremMixed>>,
        config: RoremMixedConfig,
    },
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &InpaintingModel) -> Result<Self> {
        match config {
            InpaintingModel::LaMa {} => {
                Ok(Self::LaMa(Arc::new(Mutex::new(LaMa::load(device).await?))))
            }
            InpaintingModel::AotInpainting {} => Ok(Self::Aot(Arc::new(Mutex::new(
                AotInpainting::load(device).await?,
            )))),
            InpaintingModel::Flux2Klein(config) => Ok(Self::Flux {
                model: Arc::new(Mutex::new(Flux2KleinInpaint::load(device).await?)),
                config: config.clone(),
            }),
            InpaintingModel::RoremMixed(config) => Ok(Self::Rorem {
                model: Arc::new(Mutex::new(RoremMixed::load(device).await?)),
                config: config.clone(),
            }),
        }
    }

    async fn run(
        &self,
        input: StageInput,
        seam_safe: bool,
        debug_mask_dir: Option<&Path>,
    ) -> Result<koharu_scene::Patch> {
        let mut prepared = prepare(&input, debug_mask_dir)?;
        if prepared.mask.as_raw().iter().all(|value| *value == 0) {
            return finish(input.scene.edit());
        }
        let mask = prepared.mask.clone();
        let original = prepared.original.clone();
        let cleanup = prepared.cleanup.take();
        let cleanup_entity = prepared.cleanup_entity;
        let (model_name, image) = match self {
            Self::LaMa(model) => {
                let model = model.clone();
                (
                    "lama",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("LaMa model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            prepared.rescue_flat.as_ref(),
                            seam_safe,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(
                                    image,
                                    mask,
                                    &InpaintRequest::default(),
                                )?))
                            },
                        )
                    })
                    .await
                    .context("LaMa task panicked")??,
                )
            }
            Self::Aot(model) => {
                let model = model.clone();
                (
                    "aot-inpainting",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("AOT model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            prepared.rescue_flat.as_ref(),
                            seam_safe,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(image, mask)?))
                            },
                        )
                    })
                    .await
                    .context("AOT task panicked")??,
                )
            }
            Self::Flux { model, config } => {
                let model = model.clone();
                let config = config.clone();
                (
                    "flux2-klein",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("FLUX model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            prepared.rescue_flat.as_ref(),
                            seam_safe,
                            |image, mask| {
                                model.inference(
                                    &config.prompt,
                                    image,
                                    None,
                                    &DynamicImage::ImageLuma8(mask.clone()),
                                    &Flux2KleinInpaintOptions {
                                        strength: config.strength,
                                        num_inference_steps: config.num_inference_steps,
                                        seed: config.seed,
                                        ..Flux2KleinInpaintOptions::default()
                                    },
                                )
                            },
                        )
                    })
                    .await
                    .context("FLUX task panicked")??,
                )
            }
            Self::Rorem { model, config } => {
                let model = model.clone();
                let config = config.clone();
                (
                    "rorem-mixed",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("RORem model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            prepared.rescue_flat.as_ref(),
                            seam_safe,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(
                                    image,
                                    mask,
                                    &config.prompt,
                                    &config.negative_prompt,
                                    &RoremMixedOptions {
                                        resolution: config.resolution,
                                        num_inference_steps: config.num_inference_steps,
                                        seed: config.seed,
                                        ..RoremMixedOptions::default()
                                    },
                                )?))
                            },
                        )
                    })
                    .await
                    .context("RORem task panicked")??,
                )
            }
        };
        let page = input.page;
        let manual = input.inpainting_mask.is_some();
        let mut edit = if manual {
            input.scene.edit()
        } else {
            input.scene.edit_as(generation(PRODUCER, model_name)?)
        };
        edit.observe_assets(page)?;
        if let Some(entity) = cleanup_entity {
            edit.observe::<RasterLayer>(entity)?;
            edit.observe_assets(entity)?;
        }
        let image = image.to_rgba8();
        if image.dimensions() != original.dimensions() || image.dimensions() != mask.dimensions() {
            bail!("inpainted image dimensions do not match page {page}");
        }
        let original = original.to_rgba8();
        let mut overlay = if manual {
            cleanup.unwrap_or_else(|| RgbaImage::new(image.width(), image.height()))
        } else {
            RgbaImage::new(image.width(), image.height())
        };
        for (x, y, target) in overlay.enumerate_pixels_mut() {
            if mask.get_pixel(x, y)[0] < 127 {
                continue;
            }
            let generated = image.get_pixel(x, y);
            let source = original.get_pixel(x, y);
            *target = if generated.0[..3] == source.0[..3] {
                Rgba([0, 0, 0, 0])
            } else {
                Rgba([generated[0], generated[1], generated[2], 255])
            };
        }
        let mut bytes = Cursor::new(Vec::new());
        let width = overlay.width();
        let height = overlay.height();
        DynamicImage::ImageRgba8(overlay).write_to(&mut bytes, ImageFormat::Png)?;
        let cleanup_entity = if let Some(entity) = cleanup_entity {
            if manual {
                let mut layer = input
                    .scene
                    .component::<RasterLayer>(entity)?
                    .context("cleanup entity has no raster layer component")?;
                let generated = layer.origin != Origin::User
                    || input
                        .scene
                        .component::<EntityOrigin>(entity)?
                        .is_some_and(|origin| origin.origin != Origin::User);
                if generated {
                    edit.promote_entity_to_user(entity)?;
                    layer.origin = Origin::User;
                    edit.set(entity, &layer)?;
                }
            }
            entity
        } else {
            let entity = edit.add_entity(page, At::Start)?;
            edit.set(
                entity,
                &RasterLayer {
                    origin: Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: RasterLayerKind::Cleanup,
                },
            )?;
            entity
        };
        edit.set_asset(
            cleanup_entity,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(bytes.into_inner()),
                "image/png",
                AssetMetadata {
                    width: Some(width),
                    height: Some(height),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        finish(edit)
    }
}

#[derive(Clone, Debug)]
struct FlatFillRegion {
    bounds: [u32; 4],
    polygon: Vec<(f32, f32)>,
}

fn flat_fill_regions(input: &StageInput, width: u32, height: u32) -> Result<Vec<FlatFillRegion>> {
    let mut regions = Vec::new();
    for entity in input.scene.descendants(input.page)? {
        let id = entity.id();
        // Bubbles AND text regions. Balloons were once the only candidates,
        // which meant a free-standing block of text -- a caption box,
        // a sign, a VN-style narration panel -- went to the inpainter even when
        // its background was a single flat colour.
        //
        // On a 650x206 box of 8px text, measured with `clean_only`, LaMa left
        // **27% of the ink as residue and damaged 52% of the blank background**:
        // the box is wider than `TILE_SIZE`, and a tile that is almost entirely
        // masked has no context to reconstruct from, so it invents grey texture.
        // The background there is literally constant (mean 255.0, standard
        // deviation 0.00), so a flat fill is not an approximation of the right
        // answer, it IS the right answer.
        //
        // **Widening this is safe because the guard is downstream, not here.**
        // `uniform_region_color` still has to find at least
        // `UNIFORM_BACKGROUND_MIN_PIXELS` unmasked samples away from the polygon
        // edge and agree to within 7-10 per channel. Text over artwork fails that
        // and falls through to the inpainter exactly as before; the only new
        // behaviour is on regions whose background is provably flat.
        let eligible = input.scene.component::<Region>(id)?.is_some_and(|region| {
            region.kind == BubbleRegion::kind() || region.kind == TextRegion::kind()
        });
        if !eligible {
            continue;
        }
        let Some(geometry) = input.scene.component::<Geometry>(id)? else {
            continue;
        };
        let polygon = geometry
            .points
            .iter()
            .map(|point| (point.x as f32, point.y as f32))
            .collect::<Vec<_>>();
        if polygon.len() < 3 {
            continue;
        }
        let (mut left, mut top) = (f32::INFINITY, f32::INFINITY);
        let (mut right, mut bottom) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &(x, y) in &polygon {
            left = left.min(x);
            top = top.min(y);
            right = right.max(x);
            bottom = bottom.max(y);
        }
        let bounds = [
            left.floor().clamp(0.0, width as f32) as u32,
            top.floor().clamp(0.0, height as f32) as u32,
            right.ceil().clamp(0.0, width as f32) as u32,
            bottom.ceil().clamp(0.0, height as f32) as u32,
        ];
        if bounds[2] > bounds[0] && bounds[3] > bounds[1] {
            regions.push(FlatFillRegion { bounds, polygon });
        }
    }
    Ok(regions)
}

struct InpaintInput {
    image: Arc<DynamicImage>,
    original: Arc<DynamicImage>,
    cleanup_entity: Option<koharu_scene::EntityId>,
    cleanup: Option<RgbaImage>,
    mask: GrayImage,
    text_mask: GrayImage,
    flat_fill_regions: Vec<FlatFillRegion>,
    /// Paint-by-value RGB companion to `text-mask-rescue` -- the
    /// pixels the OCR stage proved were paper, painted directly with their
    /// colour instead of being handed to the model.
    rescue_flat: Option<RgbImage>,
}

fn prepare(input: &StageInput, debug_mask_dir: Option<&Path>) -> Result<InpaintInput> {
    let page = input.page;
    let original = input
        .images
        .get(&input.scene, page, "source")?
        .ok_or_else(|| anyhow!("page {page} has no source image"))?;
    let cleanup_entity = input.scene.children(page)?.find(|entity| {
        input
            .scene
            .component::<RasterLayer>(*entity)
            .ok()
            .flatten()
            .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
    });
    let cleanup = cleanup_entity
        .map(|entity| input.images.get(&input.scene, entity, "source"))
        .transpose()?
        .flatten()
        .map(|image| image.to_rgba8());
    if cleanup
        .as_ref()
        .is_some_and(|image| image.dimensions() != original.dimensions())
    {
        bail!("cleanup layer dimensions do not match page {page}");
    }
    let source = if input.inpainting_mask.is_some() {
        if let Some(cleanup) = cleanup.as_ref() {
            let mut composite = original.to_rgba8();
            image::imageops::overlay(&mut composite, cleanup, 0, 0);
            Arc::new(DynamicImage::ImageRgba8(composite))
        } else {
            original.clone()
        }
    } else {
        original.clone()
    };
    let mut mask = GrayImage::new(source.width(), source.height());
    let mut text_mask = GrayImage::new(source.width(), source.height());
    let mut rescue_flat: Option<RgbImage> = None;
    if let Some(transient) = &input.inpainting_mask {
        let layer = image::load_from_memory(&transient.png)?.to_luma8();
        if layer.dimensions() != mask.dimensions() {
            bail!("inpainting mask dimensions do not match page {page}");
        }
        mask = layer;
    } else {
        for role in ["text-mask", "coo-mask"] {
            if let Some(image) = input.images.get(&input.scene, page, role)? {
                let layer = image.to_luma8();
                if layer.dimensions() != mask.dimensions() {
                    bail!("{role} dimensions do not match page {page}");
                }
                for (target, source) in mask.as_mut().iter_mut().zip(layer.as_raw()) {
                    *target = (*target).max(*source);
                }
                if role == "text-mask" {
                    text_mask = layer;
                }
            }
        }
        /* THE OCR STAGE'S VETO, SUBTRACTED LAST so it beats every union above.
         *
         * `text-mask` is written during detection, before a character has been
         * read, so it carries every box the detector claimed -- including the
         * ones OCR then finds hold no text at all. Measured on three Chinese
         * webtoon slices: five birds against a sky, labelled `onomatopoeia`, read
         * as `↓ Y V √ 1`, refused for lettering by the server, and erased anyway.
         *
         * `stages::ocr::write_illegible_veto` publishes those pixels under a role
         * of its own because scene authorship forbids that stage editing
         * detection's asset. Applied to BOTH masks: `mask` decides which tiles
         * LaMa is asked for, `text_mask` decides which pixels the flat fill
         * claims, and a region spared by one but erased by the other is exactly
         * the half-fixed state this change exists to remove.
         *
         * Deliberately NOT applied to the transient branch above: that mask is a
         * cleanup layer the editor authored, and someone who draws a mask by hand
         * means it. */
        if let Some(image) = input.images.get(&input.scene, page, "text-mask-veto")? {
            let veto = image.to_luma8();
            if veto.dimensions() != mask.dimensions() {
                bail!("text-mask-veto dimensions do not match page {page}");
            }
            for (target, spare) in mask.as_mut().iter_mut().zip(veto.as_raw()) {
                if *spare != 0 {
                    *target = 0;
                }
            }
            for (target, spare) in text_mask.as_mut().iter_mut().zip(veto.as_raw()) {
                if *spare != 0 {
                    *target = 0;
                }
            }
        }
        /* THE OCR STAGE'S RESCUE, UNIONED LAST so it beats the veto above.
         *
         * An upright-pass recovery letters a read whose box detection
         * REFUSED, so its ink was never in `text-mask` at all -- there is nothing
         * for "stop vetoing" to restore, and without this the render lettered
         * English beside a fully preserved column. The OCR stage publishes
         * the recovered boxes under a role of its own (authorship again), and
         * they join BOTH masks: `mask` so LaMa is asked for the tiles, and
         * `text_mask` so the flat fill claims the pixels where it applies. Union
         * after subtraction, so a recovery erases even where an overlapping
         * refusal is spared -- the recovery is the one with a letter to place. */
        if let Some(image) = input.images.get(&input.scene, page, "text-mask-rescue")? {
            let rescue = image.to_luma8();
            if rescue.dimensions() != mask.dimensions() {
                bail!("text-mask-rescue dimensions do not match page {page}");
            }
            for (target, erase) in mask.as_mut().iter_mut().zip(rescue.as_raw()) {
                *target = (*target).max(*erase);
            }
            for (target, erase) in text_mask.as_mut().iter_mut().zip(rescue.as_raw()) {
                *target = (*target).max(*erase);
            }
        }
        /* The rescue's FLAT-PAINT companion: a paint-by-value RGB
         * image whose non-black pixels are the paper colour to paint there
         * directly, instead of asking the model. The OCR stage proved those
         * pixels were paper in the source; the model given the same hole
         * invented a smoke cloud on two adjudicated renders, so for them the
         * flat colour IS the right answer -- `flat_fill_regions`' own
         * argument, per pixel, for a region whose hull spans two backgrounds
         * and so structurally fails the whole-region uniformity guard. */
        if let Some(image) = input.images.get(&input.scene, page, "text-mask-rescue-flat")? {
            let flat = image.to_rgb8();
            if flat.dimensions() != mask.dimensions() {
                bail!("text-mask-rescue-flat dimensions do not match page {page}");
            }
            rescue_flat = Some(flat);
        }
    }
    if let Some(bounds) = input.region {
        for (x, y, pixel) in mask.enumerate_pixels_mut() {
            if f64::from(x + 1) <= bounds.x
                || f64::from(y + 1) <= bounds.y
                || f64::from(x) >= bounds.x + bounds.width
                || f64::from(y) >= bounds.y + bounds.height
            {
                *pixel = Luma([0]);
                text_mask.put_pixel(x, y, Luma([0]));
            }
        }
    }
    let flat_fill_regions = flat_fill_regions(input, source.width(), source.height())?;
    // What the eraser was actually given. Cheap beside a LaMa inference, and it
    // is the number that was missing when `clean_only` silently stopped erasing:
    // the stage was admitted, profiled and reported as running, and nothing
    // anywhere said whether the mask it got was empty.
    tracing::info!(
        target: "koharu_pipeline::inpainting",
        mask_px = mask.as_raw().iter().filter(|value| **value >= 127).count(),
        text_mask_px = text_mask.as_raw().iter().filter(|value| **value >= 127).count(),
        flat_fill = flat_fill_regions.len(),
        rescue_flat_px = rescue_flat
            .as_ref()
            .map_or(0, |flat| flat.pixels().filter(|pixel| pixel.0 != [0, 0, 0]).count()),
        "inpainting mask prepared"
    );
    /* The mask the model is ACTUALLY handed -- post illegibility veto, post
     * rescue, post region clip -- which is a later, different object than the
     * detection stage's `text-mask` asset (that one predates OCR entirely).
     * Debug-only; a failed write warns and the run continues. */
    if let Some(dir) = debug_mask_dir {
        let dump = || -> Result<()> {
            std::fs::create_dir_all(dir)?;
            let tag: String = page
                .to_string()
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
                .collect();
            mask.save(dir.join(format!("p{tag}_inpaint-mask.png")))?;
            text_mask.save(dir.join(format!("p{tag}_inpaint-text-mask.png")))?;
            Ok(())
        };
        if let Err(error) = dump() {
            tracing::warn!(%error, page = %page, "debug mask dump failed");
        }
    }
    Ok(InpaintInput {
        image: source,
        original,
        cleanup_entity,
        cleanup,
        mask,
        text_mask,
        flat_fill_regions,
        rescue_flat,
    })
}

const TILE_SIZE: u32 = 512;
const TILE_CONTEXT: u32 = 128;
const UNIFORM_BACKGROUND_MIN_PIXELS: usize = 16;

/// How close to the median a background sample counts as agreeing with it.
///
/// 8 of 255. Wide enough for scanner noise and JPEG ringing on a nominally flat
/// balloon, far narrower than the gap between paper and ink.
const FLAT_FILL_TOLERANCE: i32 = 8;

/// The share of background samples that must agree before a region is flat-filled.
///
/// 0.90 leaves room for the un-erased glyph ink that the mask misses -- the thing
/// that was refusing provably-flat boxes -- while still refusing a background
/// that is genuinely two-toned or textured, where no single value reaches 90%.
const FLAT_FILL_AGREEMENT: f64 = 0.90;

/// How far inside the polygon a pixel must sit to count as a background sample,
/// keeping the region's own outline -- a balloon's stroke, a caption box's border
/// -- out of the colour estimate.
const FLAT_FILL_EDGE_MARGIN: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InpaintTile {
    core: [u32; 4],
    crop: [u32; 4],
}

// BallonsTranslator avoids model inference for a text block when the non-text
// pixels inside its balloon are nearly uniform, and otherwise sends an enlarged
// block crop to the inpainter:
// https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/modules/inpaint/base.py#L168-L200
// Koharu uses the detected bubble polygons as those blocks. Uniform bubbles are
// filled first; only the remaining mask is split into bounded model crops.
fn inpaint_tiled(
    image: &DynamicImage,
    mask: &GrayImage,
    text_mask: &GrayImage,
    flat_fill_regions: &[FlatFillRegion],
    rescue_flat: Option<&RgbImage>,
    seam_safe: bool,
    mut inference: impl FnMut(&DynamicImage, &GrayImage) -> Result<DynamicImage>,
) -> Result<DynamicImage> {
    ensure!(
        image.dimensions() == mask.dimensions() && mask.dimensions() == text_mask.dimensions(),
        "image and mask dimensions differ: image={:?}, mask={:?}, text_mask={:?}",
        image.dimensions(),
        mask.dimensions(),
        text_mask.dimensions()
    );

    let mut output = image.to_rgb8();
    let mut pending_mask = mask.clone();
    fill_uniform_regions(&mut output, &mut pending_mask, text_mask, flat_fill_regions);
    if let Some(flat) = rescue_flat {
        paint_rescue_flat(&mut output, &mut pending_mask, flat);
    }
    for tile in inpaint_tiles(&pending_mask) {
        let [left, top, right, bottom] = tile.crop;
        let crop_width = right - left;
        let crop_height = bottom - top;
        let crop_image =
            image::imageops::crop_imm(&output, left, top, crop_width, crop_height).to_image();
        let crop_mask = crop_tile_mask(&pending_mask, tile, seam_safe);

        let generated = inference(&DynamicImage::ImageRgb8(crop_image), &crop_mask)?;
        let generated = if generated.dimensions() == (crop_width, crop_height) {
            generated.to_rgb8()
        } else {
            generated
                .resize_exact(
                    crop_width,
                    crop_height,
                    image::imageops::FilterType::Lanczos3,
                )
                .to_rgb8()
        };
        composite_generated(&mut output, &pending_mask, tile, &generated);
    }
    Ok(DynamicImage::ImageRgb8(output))
}

/// The rescue's flat paint: where the rescue's companion image carries a
/// colour (non-black) AND the pixel is still pending, paint it and retire it
/// from the model's mask. Keyed on `pending_mask` so the region clip and the
/// veto keep their authority -- a clipped pixel is never painted.
fn paint_rescue_flat(output: &mut RgbImage, pending_mask: &mut GrayImage, flat: &RgbImage) {
    if flat.dimensions() != pending_mask.dimensions() {
        return;
    }
    for (x, y, colour) in flat.enumerate_pixels() {
        if colour.0 == [0, 0, 0] {
            continue;
        }
        let pending = &mut pending_mask.get_pixel_mut(x, y).0[0];
        if *pending >= 127 {
            *pending = 0;
            output.put_pixel(x, y, *colour);
        }
    }
}

fn fill_uniform_regions(
    output: &mut RgbImage,
    pending_mask: &mut GrayImage,
    text_mask: &GrayImage,
    regions: &[FlatFillRegion],
) {
    for region in regions {
        let [left, top, right, bottom] = region.bounds;
        let mut targets = Vec::new();
        for y in top..bottom {
            for x in left..right {
                if pending_mask.get_pixel(x, y)[0] >= 127
                    && text_mask.get_pixel(x, y)[0] >= 127
                    && point_in_polygon((x as f32 + 0.5, y as f32 + 0.5), &region.polygon)
                {
                    targets.push((x, y));
                }
            }
        }
        if targets.is_empty() {
            continue;
        }
        let Some(color) = uniform_region_color(output, pending_mask, region) else {
            continue;
        };
        for (x, y) in targets {
            output.put_pixel(x, y, color);
            pending_mask.put_pixel(x, y, Luma([0]));
        }
    }
}

fn inpaint_tiles(mask: &GrayImage) -> Vec<InpaintTile> {
    let mut tiles = Vec::new();
    let mut top = 0;
    while top < mask.height() {
        let bottom = top.saturating_add(TILE_SIZE).min(mask.height());
        let mut left = 0;
        while left < mask.width() {
            let right = left.saturating_add(TILE_SIZE).min(mask.width());
            if let Some([mask_left, mask_top, mask_right, mask_bottom]) =
                mask_bounds_in(mask, [left, top, right, bottom])
            {
                tiles.push(InpaintTile {
                    core: [left, top, right, bottom],
                    crop: [
                        mask_left.saturating_sub(TILE_CONTEXT),
                        mask_top.saturating_sub(TILE_CONTEXT),
                        mask_right.saturating_add(TILE_CONTEXT).min(mask.width()),
                        mask_bottom.saturating_add(TILE_CONTEXT).min(mask.height()),
                    ],
                });
            }
            left = right;
        }
        top = bottom;
    }
    tiles
}

/// The mask handed to the model for one tile.
///
/// **`seam_safe` decides whether a glyph cut by the tile grid reaches the model
/// whole, and it is the difference between erasing a drawn effect and smearing
/// it.**
///
/// The crop is `mask_bounds(core) ± TILE_CONTEXT`, so when a mask component
/// crosses a 512 grid line the half belonging to the *neighbouring* core is
/// present in this crop. Marking only the core leaves that half black — i.e. it
/// is handed to LaMa as real, known-good image, touching the hole. LaMa does the
/// reasonable thing and continues the stroke inward. `composite_generated` then
/// keeps the continuation wherever the page mask is set, and the next tile reads
/// those already-hallucinated pixels back out of `output` and continues them
/// again.
///
/// Measured on one test webtoon page, residue as a share of the
/// effect's own ink, by distance to the nearest 512 grid line: **42.5% at 0-32px,
/// 45.9% at 32-64, 15.5% at 64-128, and 1.4% beyond 128.** The cliff sits exactly
/// at `TILE_CONTEXT`, which is the reach of the mechanism.
///
/// **The growth is restricted to the core mask's own 8-connected component, and
/// that restriction is the whole design rather than a refinement.** Marking every
/// mask pixel in the crop was measured and is much worse: on that page it newly hides
/// 353,164 px of which only 11.4% is un-erased effect ink — 49.5% is a previous
/// tile's finished inpainting, read back out of `output` as context, and 43.8% is
/// untouched artwork. That is 8.7:1 collateral, and it pushes four of five tiles
/// on that page past 65% masked, where `inpaint_tile` already records that LaMa
/// "has no context to reconstruct from, so it invents grey texture".
///
/// **It cannot erase one pixel of artwork the current mask does not already
/// erase.** The flood only sets pixels that are already `>= 127` in the page
/// mask, and `composite_generated` writes back only inside `core` and only where
/// the page mask is set — so the page's erased footprint is bit-identical with
/// this on or off. What changes is how well the pixels already inside the mask
/// are reconstructed.
fn crop_tile_mask(mask: &GrayImage, tile: InpaintTile, seam_safe: bool) -> GrayImage {
    let [left, top, right, bottom] = tile.crop;
    let [core_left, core_top, core_right, core_bottom] = tile.core;
    let mut crop = GrayImage::new(right - left, bottom - top);
    let mut frontier = Vec::new();
    for y in core_top..core_bottom {
        for x in core_left..core_right {
            if mask.get_pixel(x, y)[0] >= 127 {
                crop.put_pixel(x - left, y - top, Luma([u8::MAX]));
                if seam_safe {
                    frontier.push((x, y));
                }
            }
        }
    }
    // 8-connected flood outward from what the core already claimed, clipped to
    // the crop. Everything it can reach is by construction part of the same blob,
    // so an unrelated region sitting in the context ring keeps its pixels and
    // stays available to the model as context.
    while let Some((x, y)) = frontier.pop() {
        for (nx, ny) in [
            (x.wrapping_sub(1), y.wrapping_sub(1)),
            (x, y.wrapping_sub(1)),
            (x + 1, y.wrapping_sub(1)),
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x.wrapping_sub(1), y + 1),
            (x, y + 1),
            (x + 1, y + 1),
        ] {
            if nx < left || nx >= right || ny < top || ny >= bottom {
                continue;
            }
            if crop.get_pixel(nx - left, ny - top)[0] != 0 {
                continue;
            }
            if mask.get_pixel(nx, ny)[0] < 127 {
                continue;
            }
            crop.put_pixel(nx - left, ny - top, Luma([u8::MAX]));
            frontier.push((nx, ny));
        }
    }
    crop
}

fn uniform_region_color(
    image: &RgbImage,
    mask: &GrayImage,
    region: &FlatFillRegion,
) -> Option<Rgb<u8>> {
    let [left, top, right, bottom] = region.bounds;
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for y in top..bottom {
        for x in left..right {
            let point = (x as f32 + 0.5, y as f32 + 0.5);
            if mask.get_pixel(x, y)[0] >= 127
                || !point_in_polygon(point, &region.polygon)
                || polygon_edge_distance_squared(point, &region.polygon)
                    < FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN
            {
                continue;
            }
            let pixel = image.get_pixel(x, y);
            for channel in 0..3 {
                channels[channel].push(pixel[channel]);
            }
        }
    }
    if channels[0].len() < UNIFORM_BACKGROUND_MIN_PIXELS {
        return None;
    }

    let medians = channels.each_mut().map(|values| median(values));

    // A SUPERMAJORITY near the median, not a standard deviation.
    //
    // The samples are meant to be background, but the erase mask under-covers the
    // glyphs, so a few pixels of un-erased ink get sampled as background too --
    // and a standard deviation is not robust to that. Measured on a 650x206 box
    // of 12px text whose background is a literal constant 255: 83 samples, median
    // 255, and sigma 28.0 against a threshold of 10, from a handful of black
    // outliers. The region was refused and handed to LaMa, which smeared it.
    //
    // That failure is self-reinforcing, which is why it is worth fixing rather
    // than tuning around: the pages whose masks are worst produce the most
    // outliers, so exactly the regions that most need a flat fill are the ones
    // the test rejects.
    //
    // A supermajority tolerates a minority of un-erased ink while still refusing
    // a genuinely textured background, where no single value can command
    // `FLAT_FILL_AGREEMENT` of the samples. That second half is the part that
    // keeps this safe: it is a stricter statement than "sigma is small", not a
    // looser one -- a bimodal half-flat-half-artwork region passes the old test
    // more easily than this one.
    let agreement = std::array::from_fn::<_, 3, _>(|channel| {
        let median = i32::from(medians[channel]);
        let near = channels[channel]
            .iter()
            .filter(|value| (i32::from(**value) - median).abs() <= FLAT_FILL_TOLERANCE)
            .count();
        near as f64 / channels[channel].len() as f64
    });
    (agreement.iter().copied().fold(1.0, f64::min) >= FLAT_FILL_AGREEMENT).then_some(Rgb(medians))
}

fn point_in_polygon(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for &current in polygon {
        if (current.1 > point.1) != (previous.1 > point.1) {
            let intersection_x = (previous.0 - current.0) * (point.1 - current.1)
                / (previous.1 - current.1)
                + current.0;
            if point.0 < intersection_x {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

fn polygon_edge_distance_squared(point: (f32, f32), polygon: &[(f32, f32)]) -> f32 {
    let mut minimum = f32::INFINITY;
    let mut start = polygon[polygon.len() - 1];
    for &end in polygon {
        let segment = (end.0 - start.0, end.1 - start.1);
        let length_squared = segment.0 * segment.0 + segment.1 * segment.1;
        let projection = if length_squared > 0.0 {
            ((point.0 - start.0) * segment.0 + (point.1 - start.1) * segment.1) / length_squared
        } else {
            0.0
        }
        .clamp(0.0, 1.0);
        let closest = (
            start.0 + segment.0 * projection,
            start.1 + segment.1 * projection,
        );
        let dx = point.0 - closest.0;
        let dy = point.1 - closest.1;
        minimum = minimum.min(dx * dx + dy * dy);
        start = end;
    }
    minimum
}

fn mask_bounds_in(
    mask: &GrayImage,
    [region_left, region_top, region_right, region_bottom]: [u32; 4],
) -> Option<[u32; 4]> {
    let mut left = region_right;
    let mut top = region_bottom;
    let mut right = 0;
    let mut bottom = 0;
    for y in region_top..region_bottom {
        for x in region_left..region_right {
            if mask.get_pixel(x, y)[0] >= 127 {
                left = left.min(x);
                top = top.min(y);
                right = right.max(x + 1);
                bottom = bottom.max(y + 1);
            }
        }
    }
    (right > left && bottom > top).then_some([left, top, right, bottom])
}

fn median(values: &mut [u8]) -> u8 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        ((u16::from(values[middle - 1]) + u16::from(values[middle])) / 2) as u8
    } else {
        values[middle]
    }
}

fn composite_generated(
    output: &mut RgbImage,
    mask: &GrayImage,
    tile: InpaintTile,
    generated: &RgbImage,
) {
    let [left, top, _, _] = tile.crop;
    let [core_left, core_top, core_right, core_bottom] = tile.core;
    for y in core_top..core_bottom {
        for x in core_left..core_right {
            if mask.get_pixel(x, y)[0] >= 127 {
                output.put_pixel(x, y, *generated.get_pixel(x - left, y - top));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_inpainting_mask_replaces_persistent_page_masks() {
        let mut session = koharu_scene::Session::memory().unwrap();
        let mut page = None;
        let source = DynamicImage::new_rgb8(8, 8);
        let persistent = DynamicImage::ImageLuma8(GrayImage::from_pixel(8, 8, Luma([255])));
        let encode = |image: &DynamicImage| {
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, ImageFormat::Png).unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                for (role, image) in [("source", &source), ("text-mask", &persistent)] {
                    edit.set_asset(
                        id,
                        &AssetRole::new(role)?,
                        AssetInput::new(
                            encode(image),
                            "image/png",
                            AssetMetadata {
                                width: Some(8),
                                height: Some(8),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).unwrap().snapshot;
        let page = page.unwrap();
        let mut transient = GrayImage::new(8, 8);
        transient.put_pixel(3, 4, Luma([255]));
        let input = StageInput::new(
            snapshot,
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            Some(crate::InpaintingMask {
                page,
                png: encode(&DynamicImage::ImageLuma8(transient)),
            }),
            // Story context. A `StageInput::new` parameter not threaded through
            // here stops this whole test target compiling -- which silently
            // takes the detection tests down with it.
            Arc::from(Vec::new()),
            // No glossary either -- same threading note as above.
            Arc::from(Vec::new()),
            // No re-roll seed -- same threading note as above.
            None,
            // No reader edits -- same threading note as above.
            Arc::from(Vec::new()),
            Arc::from(Vec::new()),
            // Not a joined page. Threaded here deliberately rather than left to
            // the next person, for exactly the reason the note above records.
            false,
            Arc::from([] as [f64; 0]),
            None,
        );

        let prepared = prepare(&input, None).unwrap();
        assert_eq!(prepared.mask.get_pixel(3, 4), &Luma([255]));
        assert_eq!(prepared.mask.get_pixel(0, 0), &Luma([0]));
        assert!(prepared.text_mask.pixels().all(|pixel| pixel[0] == 0));
    }

    /// The composed mask arithmetic the reader actually gets:
    /// `(text-mask − veto) ∪ rescue`, asserted through `prepare` itself rather
    /// than on either half -- a union written before the subtraction would pass
    /// any half-test and still leave the recovered column standing.
    #[test]
    fn a_rescued_recovery_erases_even_where_the_veto_spares() {
        let mut session = koharu_scene::Session::memory().unwrap();
        let mut page = None;
        let source = DynamicImage::new_rgb8(8, 8);
        // text-mask claims x<4; the veto spares x in 2..6; the rescue erases x>=5.
        let mut detection = GrayImage::new(8, 8);
        let mut veto = GrayImage::new(8, 8);
        let mut rescue = GrayImage::new(8, 8);
        for y in 0..8 {
            for x in 0..4 {
                detection.put_pixel(x, y, Luma([255]));
            }
            for x in 2..6 {
                veto.put_pixel(x, y, Luma([255]));
            }
            for x in 5..8 {
                rescue.put_pixel(x, y, Luma([255]));
            }
        }
        let encode = |image: &DynamicImage| {
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, ImageFormat::Png).unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        let layers: [(&str, DynamicImage); 4] = [
            ("source", source),
            ("text-mask", DynamicImage::ImageLuma8(detection)),
            ("text-mask-veto", DynamicImage::ImageLuma8(veto)),
            ("text-mask-rescue", DynamicImage::ImageLuma8(rescue)),
        ];
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                for (role, image) in &layers {
                    edit.set_asset(
                        id,
                        &AssetRole::new(*role)?,
                        AssetInput::new(
                            encode(image),
                            "image/png",
                            AssetMetadata {
                                width: Some(8),
                                height: Some(8),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).unwrap().snapshot;
        let input = StageInput::new(
            snapshot,
            page.unwrap(),
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            None,
            Arc::from(Vec::new()),
            Arc::from(Vec::new()),
            None,
            Arc::from(Vec::new()),
            Arc::from(Vec::new()),
            false,
            Arc::from([] as [f64; 0]),
            None,
        );

        let prepared = prepare(&input, None).unwrap();
        for (mask, name) in [(&prepared.mask, "mask"), (&prepared.text_mask, "text_mask")] {
            // x=1: detection only -- erased, exactly as before this change.
            assert_eq!(mask.get_pixel(1, 3), &Luma([255]), "{name}: detection-only pixel");
            // x=3: detection minus veto -- spared, the veto still works.
            assert_eq!(mask.get_pixel(3, 3), &Luma([0]), "{name}: vetoed pixel stays spared");
            // x=5: veto AND rescue -- ERASED: the union is after the subtraction,
            // or the recovered column is left standing beside its English.
            assert_eq!(mask.get_pixel(5, 3), &Luma([255]), "{name}: rescue beats the veto");
            // x=7: rescue alone, outside detection's mask entirely -- ERASED:
            // the whole point, ink detection never masked joins the erase.
            assert_eq!(mask.get_pixel(7, 3), &Luma([255]), "{name}: rescue adds new ink");
        }
    }

    fn rectangle_region([left, top, right, bottom]: [u32; 4]) -> FlatFillRegion {
        FlatFillRegion {
            bounds: [left, top, right, bottom],
            polygon: vec![
                (left as f32, top as f32),
                (right as f32, top as f32),
                (right as f32, bottom as f32),
                (left as f32, bottom as f32),
            ],
        }
    }

    #[test]
    fn rescue_flat_paints_only_pending_nonblack_pixels_and_retires_them() {
        let mut output = RgbImage::from_pixel(8, 8, Rgb([10, 10, 10]));
        let mut pending = GrayImage::new(8, 8);
        pending.put_pixel(2, 2, Luma([u8::MAX]));
        pending.put_pixel(3, 3, Luma([u8::MAX]));
        let mut flat = RgbImage::new(8, 8);
        flat.put_pixel(2, 2, Rgb([242, 240, 240]));
        flat.put_pixel(5, 5, Rgb([242, 240, 240])); // not pending: never painted
        paint_rescue_flat(&mut output, &mut pending, &flat);
        assert_eq!(output.get_pixel(2, 2), &Rgb([242, 240, 240]), "painted");
        assert_eq!(pending.get_pixel(2, 2), &Luma([0]), "retired from the model");
        assert_eq!(output.get_pixel(5, 5), &Rgb([10, 10, 10]), "not pending, untouched");
        assert_eq!(
            pending.get_pixel(3, 3),
            &Luma([u8::MAX]),
            "pending without a colour stays the model's"
        );
        assert_eq!(output.get_pixel(3, 3), &Rgb([10, 10, 10]));
    }

    #[test]
    fn uniform_mask_background_is_filled_without_inference() {
        let mut image = RgbImage::from_pixel(96, 96, Rgb([240, 241, 242]));
        let mut mask = GrayImage::new(96, 96);
        for y in 40..56 {
            for x in 32..64 {
                image.put_pixel(x, y, Rgb([20, 20, 20]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[rectangle_region([0, 0, 96, 96])],
            None,
            false,
            |_, _| {
                calls += 1;
                Ok(DynamicImage::new_rgb8(1, 1))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 0);
        assert_eq!(output.get_pixel(48, 48), &Rgb([240, 241, 242]));
    }

    /// The regression that motivated the supermajority test. The erase mask
    /// under-covers the glyphs, so a MINORITY of un-erased ink is sampled as
    /// background. A standard deviation is not robust to that: measured on a real
    /// 650x206 box of 12px text whose background is a literal constant 255, 83
    /// samples with median 255 gave sigma 28.0 against a threshold of 10, and the
    /// region was refused and handed to LaMa, which smeared it.
    #[test]
    fn a_minority_of_unerased_ink_does_not_refuse_a_flat_background() {
        let mut image = RgbImage::from_pixel(96, 96, Rgb([255, 255, 255]));
        let mut mask = GrayImage::new(96, 96);
        for y in 40..56 {
            for x in 32..64 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        // Ink the mask missed: black, unmasked, and therefore sampled as
        // "background". Well under a tenth of the samples.
        for x in 10..24 {
            image.put_pixel(x, 10, Rgb([0, 0, 0]));
        }

        let region = rectangle_region([0, 0, 96, 96]);
        assert_eq!(
            uniform_region_color(&image, &mask, &region),
            Some(Rgb([255, 255, 255])),
            "a flat white background must survive a minority of un-erased ink"
        );
    }

    /// The other half of the same rule, and the reason it is a SUPERMAJORITY
    /// rather than a looser sigma: a background that is genuinely two-toned must
    /// still be refused, so that half a region of artwork is never painted over
    /// with the median of the other half.
    #[test]
    fn a_two_toned_background_is_still_refused() {
        let mut image = RgbImage::from_pixel(96, 96, Rgb([255, 255, 255]));
        let mask = GrayImage::new(96, 96);
        for y in 0..96 {
            for x in 48..96 {
                image.put_pixel(x, y, Rgb([90, 90, 90]));
            }
        }
        assert_eq!(
            uniform_region_color(&image, &mask, &rectangle_region([0, 0, 96, 96])),
            None,
            "half flat white and half flat grey is not a uniform background"
        );
    }

    #[test]
    fn uniform_bubbles_are_filled_independently_inside_one_textured_tile() {
        let mut image = RgbImage::new(200, 100);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([20, 80, 140])
            } else {
                Rgb([220, 80, 120])
            };
        }
        for y in 10..90 {
            for x in 10..90 {
                image.put_pixel(x, y, Rgb([245, 245, 245]));
            }
            for x in 110..190 {
                image.put_pixel(x, y, Rgb([250, 240, 220]));
            }
        }
        let mut mask = GrayImage::new(200, 100);
        for y in 40..60 {
            for x in 30..60 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
            for x in 135..165 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[
                rectangle_region([10, 10, 90, 90]),
                rectangle_region([110, 10, 190, 90]),
            ],
            None,
            false,
            |_, _| {
                calls += 1;
                Ok(DynamicImage::new_rgb8(1, 1))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 0);
        assert_eq!(output.get_pixel(45, 50), &Rgb([245, 245, 245]));
        assert_eq!(output.get_pixel(150, 50), &Rgb([250, 240, 220]));
    }

    #[test]
    fn textured_mask_regions_are_inferred_as_bounded_tiles() {
        let mut image = RgbImage::new(1200, 700);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([0, 40, 80])
            } else {
                Rgb([255, 215, 175])
            };
        }
        let original = image.clone();
        let mut mask = GrayImage::new(1200, 700);
        mask.put_pixel(10, 10, Luma([u8::MAX]));
        mask.put_pixel(1100, 600, Luma([u8::MAX]));
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[],
            None,
            false,
            |tile: &DynamicImage, _: &GrayImage| {
                calls += 1;
                assert!(tile.width() <= TILE_CONTEXT * 2 + 1);
                assert!(tile.height() <= TILE_CONTEXT * 2 + 1);
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([1, 2, 3]),
                )))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 2);
        assert_eq!(output.get_pixel(10, 10), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(1100, 600), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(500, 300), original.get_pixel(500, 300));
    }

    /// A blob straddling the 512 grid line reaches the model whole.
    ///
    /// **Without this the half in the neighbouring core arrives as unmasked
    /// context touching the hole** -- LaMa reads it as real image and continues
    /// the stroke inward, which is where the residue on a drawn sound effect
    /// comes from. Asserted on the crop mask, because that is where the decision
    /// is made.
    #[test]
    fn a_blob_cut_by_the_tile_grid_is_masked_whole_in_both_crops() {
        let mut mask = GrayImage::new(1024, 256);
        for y in 100..160 {
            for x in 400..700 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let tiles = inpaint_tiles(&mask);
        assert_eq!(tiles.len(), 2, "the blob should straddle two cores");
        for tile in tiles {
            let [left, top, _, _] = tile.crop;
            let off = crop_tile_mask(&mask, tile, false);
            let on = crop_tile_mask(&mask, tile, true);
            let count = |m: &GrayImage| m.pixels().filter(|p| p.0[0] != 0).count();
            assert!(count(&on) > count(&off), "must reach the other core's half");
            for y in 100..160 {
                for x in 400..700 {
                    let cx = x as i64 - i64::from(left);
                    let cy = y as i64 - i64::from(top);
                    if cx < 0 || cy < 0 {
                        continue;
                    }
                    let (cx, cy) = (cx as u32, cy as u32);
                    if cx >= on.width() || cy >= on.height() {
                        continue;
                    }
                    assert_eq!(on.get_pixel(cx, cy).0[0], u8::MAX, "({x},{y}) unmasked");
                }
            }
        }
    }

    /// The flood follows ONE component and stops. An unrelated masked region in
    /// the same context ring keeps its pixels and stays available as context --
    /// marking the whole crop instead was measured at 8.7:1 collateral, and this
    /// is the assertion that keeps the two designs apart.
    #[test]
    fn an_unrelated_region_in_the_context_ring_is_left_alone() {
        let mut mask = GrayImage::new(1024, 256);
        for y in 100..160 {
            for x in 480..560 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        for y in 100..160 {
            for x in 600..640 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let first = inpaint_tiles(&mask)
            .into_iter()
            .find(|t| t.core[0] == 0)
            .expect("a tile whose core starts at x=0");
        let [left, top, _, _] = first.crop;
        let on = crop_tile_mask(&mask, first, true);
        assert_eq!(on.get_pixel(555 - left, 130 - top).0[0], u8::MAX);
        assert_eq!(on.get_pixel(620 - left, 130 - top).0[0], 0);
    }

    /// The off arm is today's function, op for op. The A/B depends on it.
    #[test]
    fn the_off_arm_marks_only_the_core() {
        let mut mask = GrayImage::new(1024, 256);
        for y in 100..160 {
            for x in 480..560 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let first = inpaint_tiles(&mask)
            .into_iter()
            .find(|t| t.core[0] == 0)
            .expect("a tile whose core starts at x=0");
        let [left, top, _, _] = first.crop;
        let off = crop_tile_mask(&mask, first, false);
        assert_eq!(off.get_pixel(500 - left, 130 - top).0[0], u8::MAX);
        assert_eq!(off.get_pixel(555 - left, 130 - top).0[0], 0);
    }
}
