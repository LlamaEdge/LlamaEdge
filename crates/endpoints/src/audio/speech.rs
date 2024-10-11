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
    pub voice: SpeechVoice,
    /// The format to audio in. Supported formats are `mp3`, `opus`, `aac`, `flac`, `wav`, and `pcm`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<SpeechFormat>,
    /// The speed of the generated audio. Select a value from `0.25` to `4.0`. `1.0` is the default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
}

impl<'de> Deserialize<'de> for SpeechRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Model,
            Input,
            Voice,
            ResponseFormat,
            Speed,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl Visitor<'_> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter
                            .write_str("`model`, `input`, `voice`, `response_format`, or `speed`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "model" => Ok(Field::Model),
                            "input" => Ok(Field::Input),
                            "voice" => Ok(Field::Voice),
                            "response_format" => Ok(Field::ResponseFormat),
                            "speed" => Ok(Field::Speed),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

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

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Model => {
                            if model.is_some() {
                                return Err(de::Error::duplicate_field("model"));
                            }
                            model = Some(map.next_value()?);
                        }
                        Field::Input => {
                            if input.is_some() {
                                return Err(de::Error::duplicate_field("input"));
                            }
                            input = Some(map.next_value()?);
                        }
                        Field::Voice => {
                            if voice.is_some() {
                                return Err(de::Error::duplicate_field("voice"));
                            }
                            voice = Some(map.next_value()?);
                        }
                        Field::ResponseFormat => {
                            response_format = map.next_value()?;
                        }
                        Field::Speed => {
                            speed = map.next_value()?;
                        }
                    }
                }

                let model = model.ok_or_else(|| de::Error::missing_field("model"))?;
                let input = input.ok_or_else(|| de::Error::missing_field("input"))?;
                let voice = voice.ok_or_else(|| de::Error::missing_field("voice"))?;
                if response_format.is_none() {
                    response_format = Some(SpeechFormat::Wav);
                }
                if speed.is_none() {
                    speed = Some(1.0);
                }

                Ok(SpeechRequest {
                    model,
                    input,
                    voice,
                    response_format,
                    speed,
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
        assert_eq!(speech_request.voice, SpeechVoice::Alloy);
        assert_eq!(speech_request.response_format, Some(SpeechFormat::Wav));
        assert_eq!(speech_request.speed, Some(1.0));
    }

    {
        let json = r#"{
            "model": "test_model",
            "input": "This is an input",
            "voice": "alloy",
            "response_format": "wav",
            "speed": 1.5
        }"#;
        let speech_request: SpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(speech_request.model, "test_model");
        assert_eq!(speech_request.input, "This is an input");
        assert_eq!(speech_request.voice, SpeechVoice::Alloy);
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
