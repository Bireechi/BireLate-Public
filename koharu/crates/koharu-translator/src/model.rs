use koharu_ml::llm::GenerationOptions;
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::Provider;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default, deny_unknown_fields)]
pub struct GenerationConfig {
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub thinking: bool,
    /// Whether the sliding-window layers allocate the whole context.
    ///
    /// Not a sampling knob -- it changes no logit the model would otherwise
    /// produce, only how much of the KV cache is kept. It rides here because
    /// this struct is the one channel a caller already has into the generation
    /// options, and because it is a per-deployment decision: whether it is
    /// sound depends on how the caller drives the context, not on the page.
    ///
    /// `None` leaves llama.cpp's default.
    pub swa_full: Option<bool>,
    /// The sampler seed. The caller's alone, like `swa_full` above -- no
    /// descriptor fallback, because it describes the DRAW, not the model.
    ///
    /// Upstream pins `DEFAULT_SEED` (a fixed constant, 299,792,458) and
    /// `build_sampler` runs fresh inside every `inference` call ending in
    /// `LlamaSampler::dist(options.seed)` -- so without this field every
    /// "repeat" of an identical request is byte-identical at ANY temperature,
    /// which made replay's `--repeat` a determinism check and nothing else,
    /// and made retry-on-identical-request a guaranteed no-op.
    /// `None` keeps that fixed upstream default, the shipped behaviour.
    pub seed: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct ModelSelection {
    pub provider: Provider,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
}

impl Default for ModelSelection {
    fn default() -> Self {
        Self {
            provider: Provider::Local,
            model: Some(crate::local::DEFAULT_MODEL.to_owned()),
            quantization: Some(crate::local::DEFAULT_QUANTIZATION.to_owned()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Type)]
pub struct Quantization {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Type)]
pub struct Model {
    pub provider: Provider,
    pub model: Option<String>,
    pub name: String,
    pub quantizations: Vec<Quantization>,
}

impl Model {
    pub(crate) fn catalog(provider: Provider, entries: &[(&str, &str)]) -> Vec<Self> {
        entries
            .iter()
            .map(|&(model, name)| Self {
                provider,
                model: Some(model.to_owned()),
                name: name.to_owned(),
                quantizations: Vec::new(),
            })
            .collect()
    }

    pub(crate) fn service(provider: Provider, name: &str) -> Self {
        Self {
            provider,
            model: None,
            name: name.to_owned(),
            quantizations: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QuantizationDefinition {
    pub id: &'static str,
    pub name: &'static str,
    pub filename: &'static str,
}

impl QuantizationDefinition {
    pub const fn new(id: &'static str, name: &'static str, filename: &'static str) -> Self {
        Self { id, name, filename }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ModelGeneration {
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

impl ModelGeneration {
    pub(crate) fn options(self, overrides: GenerationConfig) -> GenerationOptions {
        let defaults = GenerationOptions::default();
        GenerationOptions {
            max_tokens: overrides
                .max_tokens
                .or(self.max_tokens)
                .map_or(1000, |value| value as usize),
            temperature: overrides
                .temperature
                .or(self.temperature)
                .unwrap_or(defaults.temperature),
            top_k: overrides.top_k.or(self.top_k).map(|value| value as usize),
            top_p: overrides.top_p.or(self.top_p),
            min_p: overrides.min_p.or(self.min_p),
            repeat_penalty: overrides
                .repeat_penalty
                .or(self.repeat_penalty)
                .unwrap_or(defaults.repeat_penalty),
            frequency_penalty: overrides
                .frequency_penalty
                .or(self.frequency_penalty)
                .unwrap_or(defaults.frequency_penalty),
            presence_penalty: overrides
                .presence_penalty
                .or(self.presence_penalty)
                .unwrap_or(defaults.presence_penalty),
            /* The caller's alone -- deliberately no descriptor fallback. Every
             * other field here falls back to the model catalog because it
             * describes the MODEL; this one describes how the process drives the
             * context, which the catalog cannot know and must not decide. */
            swa_full: overrides.swa_full,
            seed: overrides.seed.unwrap_or(defaults.seed),
            ..defaults
        }
    }
}

#[must_use]
pub(crate) fn display_name(model: &str) -> String {
    model
        .rsplit_once('/')
        .map_or(model, |(_, model)| model)
        .split(['-', '_'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut characters = part.chars();
            characters.next().map_or_else(String::new, |first| {
                first.to_uppercase().chain(characters).collect()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}
