//! Define types for audio generation from the input text.

use serde::{Deserialize, Serialize};

/// Represents a request for generating audio from text.
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct SpeechRequest {
    /// Model name.
    pub model: String,
    /// The text to generate audio for.
    pub input: String,
    /// The voice to use when generating the audio. Supported voices are `alloy`, `echo`, `fable`, `onyx`, `nova`, and `shimmer`.
    pub voice: String,
    /// The format to audio in. Supported formats are `mp3`, `opus`, `aac`, `flac`, `wav`, and `pcm`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// The speed of the generated audio. Select a value from `0.25` to `4.0`. `1.0` is the default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
}
