//! PaddleOCR-VL-1.6 element recognition backed by the checkpoint revision
//! `66317acc4c9fc17bd154591ce650735cd2855f3e`.

mod config;
mod model;
mod processor;

use anyhow::{Context, Result};
use image::DynamicImage;
use koharu_torch::Device;

pub use self::{
    config::{PaddleOCRVLConfig, PaddleOCRVisionConfig, RopeScaling},
    processor::{PaddleOCRVLImageProcessor, PaddleOCRVLResult, PaddleOCRVLTask},
};

use self::{model::Model, processor::Processor};

model_repository!("PaddlePaddle/PaddleOCR-VL-1.6" @ "66317acc4c9fc17bd154591ce650735cd2855f3e" {
    CONFIG = "config.json",
    WEIGHTS = "model.safetensors",
    PROCESSOR = "preprocessor_config.json",
    TOKENIZER = "tokenizer.json",
});

#[derive(Debug)]
pub struct PaddleOCRVL {
    device: Device,
    model: Model,
    processor: Processor,
}

impl PaddleOCRVL {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let device: Device = device.try_into()?;
        let config_path = CONFIG
            .resolve()
            .await
            .context("failed to resolve PaddleOCR-VL-1.6 config")?;
        let weights_path = WEIGHTS
            .resolve()
            .await
            .context("failed to resolve PaddleOCR-VL-1.6 weights")?;
        let processor_path = PROCESSOR
            .resolve()
            .await
            .context("failed to resolve PaddleOCR-VL-1.6 image processor")?;
        let tokenizer_path = TOKENIZER
            .resolve()
            .await
            .context("failed to resolve PaddleOCR-VL-1.6 tokenizer")?;

        let config = PaddleOCRVLConfig::from_file(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let processor =
            Processor::from_files(&processor_path, &tokenizer_path, config.image_token_id)?;
        let mut model = Model::new(config, device);
        model
            .load_safetensors(&weights_path)
            .with_context(|| format!("failed to load {}", weights_path.display()))?;

        Ok(Self {
            device,
            model,
            processor,
        })
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        task: PaddleOCRVLTask,
    ) -> Result<PaddleOCRVLResult> {
        koharu_torch::no_grad(|| {
            let (pixel_values, image_grid_thw) =
                self.processor.preprocess(image, task, self.device)?;
            let (input_ids, mm_token_type_ids) =
                self.processor.encode_prompt(task, image_grid_thw)?;
            // 512 new tokens. **Do not lower this to fight a looping read**, and
            // the reason is that a loop is exactly as long as the cap.
            //
            // Measured over about 27,000 OCR reads, in characters (this
            // tokenizer is sentencepiece BPE over raw UTF-8; `ピ` and `ー` are
            // separate tokens, so a 512-character `ピー` loop on a real page
            // is exactly 512 tokens -- 1.00 chars/token for katakana, ~1.04 for
            // prose, where `今日` is one token):
            //
            //   dense fixture  truth  read   verdict
            //   wide-32           76    76   complete
            //   wide-24          130   130   complete
            //   wide-16          312   311   complete, stopped on its own
            //   wide-12          636   533   TRUNCATED BY THIS CAP
            //   wide-08         1343   497   TRUNCATED BY THIS CAP
            //   real-page loop     -   512   the defect, == the cap
            //
            // So the legitimate population already runs ABOVE the defect: the
            // empty band is negative, and any lower cap clips real fine print
            // (the longest real, non-fixture read is a 201-character school
            // notice) while merely shortening the loop it was aimed at. The
            // loop is refused instead, by content, in
            // `koharu-pipeline/src/stages/ocr.rs::degenerate_repetition`.
            let (token_ids, confidence) = self.model.forward(
                &input_ids,
                &mm_token_type_ids,
                &pixel_values,
                image_grid_thw,
                512,
            )?;
            let result = self.processor.decode(&token_ids, confidence)?;
            /* The greedy decode's length-normalised sequence probability, which the
             * model computes and this code used to discard at `argmax`.
             *
             * It does NOT separate artwork from text -- that was measured and
             * refused. It is carried on the result for a different and measured
             * reason: it ranks the two orientations of one crop, where it was right
             * or neutral on 5 of 5. The log line stays, because it is what makes an
             * A/B of that ranking falsifiable from outside the process.
             *
             * The crop's dimensions are here because the TEXT alone does not identify
             * a region -- the same string is read on many pages -- and correlating a
             * log line back to a region is the whole point of writing one. */
            tracing::debug!(
                confidence = confidence.unwrap_or(f32::NAN),
                width = image.width(),
                height = image.height(),
                chars = result.text.chars().count(),
                text = %result.text,
                "paddleocr-vl read"
            );
            Ok(result)
        })
    }
}
