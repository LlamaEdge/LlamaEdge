//! Define types for audio generation from the input text.

use serde::{
    de::{self, Deserializer, MapAccess, Visitor},
    Deserialize, Serialize,
};
use std::fmt;

/// Represents a request for generating audio from text.
#[derive(Debug, Serialize)]
pub struct SpeechRequest {
    /// Model name.
    pub model: String,
    /// The text to generate audio for.
    pub input: String,
    /// The voice to use when generating the audio. Supported voices are `alloy`, `echo`, `fable`, `onyx`, `nova`, and `shimmer`.
    pub voice: Option<SpeechVoice>,
    /// The format to audio in. Supported formats are `mp3`, `opus`, `aac`, `flac`, `wav`, and `pcm`. Defaults to `wav`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<SpeechFormat>,
    /// The speed of the generated audio. Select a value from `0.25` to `4.0`. Defaults to `1.0`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,

    /// Id of speaker. Defaults to `0`. This param is only supported for `Piper`.
    #[serde(skip_serializing_if = "Option::is_none", rename = "speaker")]
    pub speaker_id: Option<u32>,
    /// Amount of noise to add during audio generation. Defaults to `0.667`. This param is only supported for `Piper`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_scale: Option<f64>,
    /// Speed of speaking (1 = normal, < 1 is faster, > 1 is slower). Defaults to `1.0`. This param is only supported for `Piper`.
    pub length_scale: Option<f64>,
    /// Variation in phoneme lengths. Defaults to `0.8`. This param is only supported for `Piper`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_w: Option<f64>,
    /// Seconds of silence after each sentence. Defaults to `0.2`. This param is only supported for `Piper`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentence_silence: Option<f64>,
    /// Seconds of extra silence to insert after a single phoneme. This param is only supported for `Piper`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phoneme_silence: Option<f64>,
    /// stdin input is lines of JSON instead of plain text. This param is only supported for `Piper`.
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
impl<'de> Deserialize<'de> for SpeechRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SpeechRequestVisitor;

        impl<'de> Visitor<'de> for SpeechRequestVisitor {
            type Value = SpeechRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct SpeechRequest")
            }

            fn visit_map<V>(self, mut map: V) -> Result<SpeechRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut model = None;
                let mut input = None;
                let mut voice = None;
                let mut response_format = None;
                let mut speed = None;
                let mut speaker_id = None;
                let mut noise_scale = None;
                let mut noise_w = None;
                let mut sentence_silence = None;
                let mut phoneme_silence = None;
                let mut json_input = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "model" => {
                            if model.is_some() {
                                return Err(de::Error::duplicate_field("model"));
                            }
                            model = Some(map.next_value()?);
                        }
                        "input" => {
                            if input.is_some() {
                                return Err(de::Error::duplicate_field("input"));
                            }
                            input = Some(map.next_value()?);
                        }
                        "voice" => {
                            if voice.is_some() {
                                return Err(de::Error::duplicate_field("voice"));
                            }
                            voice = Some(map.next_value()?);
                        }
                        "response_format" => {
                            response_format = map.next_value()?;
                        }
                        "speed" => {
                            speed = map.next_value()?;
                        }
                        "speaker_id" => {
                            speaker_id = map.next_value()?;
                        }
                        "noise_scale" => {
                            noise_scale = map.next_value()?;
                        }
                        "noise_w" => {
                            noise_w = map.next_value()?;
                        }
                        "sentence_silence" => {
                            sentence_silence = map.next_value()?;
                        }
                        "phoneme_silence" => {
                            phoneme_silence = map.next_value()?;
                        }
                        "json_input" => {
                            json_input = map.next_value()?;
                        }
                        _ => {
                            #[cfg(feature = "logging")]
                            warn!(target: "stdout", "Not supported field: {}", key);
                        }
                    }
                }

                let model = model.ok_or_else(|| de::Error::missing_field("model"))?;

                let input = input.ok_or_else(|| de::Error::missing_field("input"))?;

                if response_format.is_none() {
                    response_format = Some(SpeechFormat::Wav);
                }

                if speed.is_none() {
                    speed = Some(1.0);
                }

                if speaker_id.is_none() {
                    speaker_id = Some(0);
                }

                if noise_scale.is_none() {
                    noise_scale = Some(0.667);
                }

                if noise_w.is_none() {
                    noise_w = Some(0.8);
                }

                if sentence_silence.is_none() {
                    sentence_silence = Some(0.2);
                }

                Ok(SpeechRequest {
                    model,
                    input,
                    voice,
                    response_format,
                    speed,
                    speaker_id,
                    noise_scale,
                    length_scale: speed,
                    noise_w,
                    sentence_silence,
                    phoneme_silence,
                    json_input,
                })
            }
        }

        const FIELDS: &[&str] = &["model", "input", "voice", "response_format", "speed"];
        deserializer.deserialize_struct("SpeechRequest", FIELDS, SpeechRequestVisitor)
    }
}

#[test]
fn test_audio_deserialize_speech_request() {
    {
        let json = r#"{
            "model": "test_model",
            "input": "This is an input",
            "voice": "alloy"
        }"#;
        let speech_request: SpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(speech_request.model, "test_model");
        assert_eq!(speech_request.input, "This is an input");
        assert_eq!(speech_request.voice, Some(SpeechVoice::Alloy));
        assert_eq!(speech_request.response_format, Some(SpeechFormat::Wav));
        assert_eq!(speech_request.speed, Some(1.0));
    }

    {
        let json = r#"{
            "model": "test_model",
            "input": "This is an input",
            "response_format": "wav",
            "speed": 1.5
        }"#;
        let speech_request: SpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(speech_request.model, "test_model");
        assert_eq!(speech_request.input, "This is an input");
        assert_eq!(speech_request.voice, None);
        assert_eq!(speech_request.response_format, Some(SpeechFormat::Wav));
        assert_eq!(speech_request.speed, Some(1.5));
    }

    {
        let json = r#"{
            "model": "test_model",
            "input": "This is an input",
            "voice": "alloy",
            "response_format": "mp3"
        }"#;
        let res: Result<SpeechRequest, serde_json::Error> = serde_json::from_str(json);
        assert!(res.is_err());
        if let Err(e) = res {
            let actual = e.to_string();
            assert!(actual.starts_with("unknown variant `mp3`, expected `wav`"));
        }
    }

    {
        let json = r#"{
            "model": "test_model",
            "input": "This is an input",
            "voice": "unknown",
        }"#;
        let res: Result<SpeechRequest, serde_json::Error> = serde_json::from_str(json);
        assert!(res.is_err());
        if let Err(e) = res {
            let actual = e.to_string();
            assert!(actual.starts_with("unknown variant `unknown`, expected one of `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`"));
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SpeechVoice {
    Alloy,
    Echo,
    Fable,
    Onyx,
    Nova,
    Shimmer,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SpeechFormat {
    Wav,
    // Mp3,
    // Opus,
    // Aac,
    // Flac,
    // Pcm,
}
