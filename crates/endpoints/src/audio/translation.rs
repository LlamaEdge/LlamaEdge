//! Define types for translating audio into English.

use crate::files::FileObject;
use serde::{Deserialize, Serialize};

/// Represents a rquest for translating audio into English.
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct TranslationRequest {
    /// The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
    pub file: FileObject,
    /// ID of the model to use.
    pub model: String,
    /// An optional text to guide the model's style or continue a previous audio segment. The prompt should be in English.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// The format of the transcript output, in one of these options: `json`, `text`, `srt`, `verbose_json`, or `vtt`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically increase the temperature until certain thresholds are hit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// Represents a translation object.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranslationObject {
    /// The translated text.
    pub text: String,
}
