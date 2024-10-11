use super::BaseMetadata;
use serde::{Deserialize, Serialize};

/// Metadata for chat and embeddings models
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PiperMetadata {
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub model_name: String,
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub model_alias: String,

    // * Plugin parameters (used by this plugin):
    #[serde(rename = "enable-debug-log")]
    pub debug_log: bool,

    /// The speed of the generated audio. Select a value from `0.25` to `4.0`. Defaults to `1.0`.
    pub speed: f64,
    /// Id of speaker. Defaults to `0`.
    pub speaker_id: u32,
    /// Amount of noise to add during audio generation. Defaults to `0.667`.
    pub noise_scale: f64,
    /// Speed of speaking (1 = normal, < 1 is faster, > 1 is slower). Defaults to `1.0`.
    length_scale: f64,
    /// Variation in phoneme lengths. Defaults to `0.8`.
    pub noise_w: f64,
    /// Seconds of silence after each sentence. Defaults to `0.2`.
    pub sentence_silence: f64,
    /// Seconds of extra silence to insert after a single phoneme.
    #[serde(skip_serializing_if = "Option::is_none")]
    phoneme_silence: Option<f64>,
    /// stdin input is lines of JSON instead of plain text.
    /// The input format should be:
    /// ```json
    /// {
    ///    "text": "some text",     (required)
    ///    "speaker_id": 0,         (optional)
    ///    "speaker": "some name",  (optional)
    /// }
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_input: Option<bool>,
}
impl Default for PiperMetadata {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_alias: String::new(),
            debug_log: false,
            speed: 1.0,
            speaker_id: 0,
            noise_scale: 0.667,
            length_scale: 1.0,
            noise_w: 0.8,
            sentence_silence: 0.2,
            phoneme_silence: None,
            json_input: None,
        }
    }
}
impl BaseMetadata for PiperMetadata {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn model_alias(&self) -> &str {
        &self.model_alias
    }
}
