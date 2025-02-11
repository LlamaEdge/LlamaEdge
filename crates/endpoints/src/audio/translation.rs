//! Define types for translating audio into English.

use crate::files::FileObject;
use serde::{
    de::{self, IgnoredAny, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use std::fmt;

/// Represents a rquest for translating audio into English.
#[derive(Debug, Serialize)]
pub struct TranslationRequest {
    /// The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
    pub file: FileObject,
    /// ID of the model to use.
    pub model: Option<String>,
    /// An optional text to guide the model's style or continue a previous audio segment. The prompt should be in English.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// The format of the transcript output, in one of these options: `json`, `text`, `srt`, `verbose_json`, or `vtt`. Defaults to `json`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically increase the temperature until certain thresholds are hit. Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// The language of the input audio. Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will improve accuracy and latency. Defaults to `en`. If `detect_language` is true, this parameter will be set to `auto`.
    pub language: Option<String>,
    /// automatically detect the spoken language in the provided audio input. Defaults to false. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detect_language: Option<bool>,
    /// Time offset in milliseconds. Defaults to 0. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_time: Option<u64>,
    /// Length of audio (in seconds) to be processed starting from the point defined by the `offset_time` field (or from the beginning by default). Defaults to 0. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u64>,
    /// Maximum amount of text context (in tokens) that the model uses when processing long audio inputs incrementally. Defaults to -1. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context: Option<i32>,
    /// Maximum number of tokens that the model can generate in a single transcription segment (or chunk). Defaults to 0. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_len: Option<u64>,
    /// Split audio chunks on word rather than on token. Defaults to false. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_on_word: Option<bool>,
}
impl Default for TranslationRequest {
    fn default() -> Self {
        TranslationRequest {
            file: FileObject::default(),
            model: None,
            prompt: None,
            response_format: Some("json".to_string()),
            temperature: Some(0.0),
            language: Some("en".to_string()),
            detect_language: Some(false),
            offset_time: Some(0),
            duration: Some(0),
            max_context: Some(-1),
            max_len: Some(0),
            split_on_word: Some(false),
        }
    }
}
impl<'de> Deserialize<'de> for TranslationRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TranslationRequestVisitor;

        impl<'de> Visitor<'de> for TranslationRequestVisitor {
            type Value = TranslationRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct TranslationRequest")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TranslationRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut file = None;
                let mut model = None;
                let mut prompt: Option<String> = None;
                let mut response_format = None;
                let mut temperature = None;
                let mut language = None;
                let mut detect_language = None;
                let mut offset_time = None;
                let mut duration = None;
                let mut max_context = None;
                let mut max_len = None;
                let mut split_on_word = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "file" => {
                            if file.is_some() {
                                return Err(de::Error::duplicate_field("file"));
                            }
                            file = Some(map.next_value()?);
                        }
                        "model" => {
                            if model.is_some() {
                                return Err(de::Error::duplicate_field("model"));
                            }
                            model = Some(map.next_value()?);
                        }
                        "prompt" => {
                            if prompt.is_some() {
                                return Err(de::Error::duplicate_field("prompt"));
                            }
                            prompt = Some(map.next_value()?);
                        }
                        "response_format" => {
                            if response_format.is_some() {
                                return Err(de::Error::duplicate_field("response_format"));
                            }
                            response_format = Some(map.next_value()?);
                        }
                        "temperature" => {
                            if temperature.is_some() {
                                return Err(de::Error::duplicate_field("temperature"));
                            }
                            temperature = Some(map.next_value()?);
                        }
                        "language" => {
                            if language.is_some() {
                                return Err(de::Error::duplicate_field("language"));
                            }
                            language = Some(map.next_value()?);
                        }
                        "detect_language" => {
                            if detect_language.is_some() {
                                return Err(de::Error::duplicate_field("detect_language"));
                            }
                            detect_language = Some(map.next_value()?);
                        }
                        "offset_time" => {
                            if offset_time.is_some() {
                                return Err(de::Error::duplicate_field("offset_time"));
                            }
                            offset_time = Some(map.next_value()?);
                        }
                        "duration" => {
                            if duration.is_some() {
                                return Err(de::Error::duplicate_field("duration"));
                            }
                            duration = Some(map.next_value()?);
                        }
                        "max_context" => {
                            if max_context.is_some() {
                                return Err(de::Error::duplicate_field("max_context"));
                            }
                            max_context = Some(map.next_value()?);
                        }
                        "max_len" => {
                            if max_len.is_some() {
                                return Err(de::Error::duplicate_field("max_len"));
                            }
                            max_len = Some(map.next_value()?);
                        }
                        "split_on_word" => {
                            if split_on_word.is_some() {
                                return Err(de::Error::duplicate_field("split_on_word"));
                            }
                            split_on_word = Some(map.next_value()?);
                        }
                        _ => {
                            // Ignore unknown fields
                            let _ = map.next_value::<IgnoredAny>()?;

                            #[cfg(feature = "logging")]
                            warn!(target: "stdout", "Not supported field: {}", key);
                        }
                    }
                }

                let file = file.ok_or_else(|| de::Error::missing_field("file"))?;

                // if detect_language is true, set language to "auto"
                match detect_language {
                    Some(true) => {
                        language = Some("auto".to_string());
                    }
                    _ => {
                        detect_language = Some(false);

                        if language.is_none() {
                            language = Some("en".to_string());
                        }
                    }
                }

                prompt = prompt.and_then(|p| if p.is_empty() { None } else { Some(p) });

                if response_format.is_none() {
                    response_format = Some("json".to_string());
                }

                if temperature.is_none() {
                    temperature = Some(0.0);
                }

                if offset_time.is_none() {
                    offset_time = Some(0);
                }

                if duration.is_none() {
                    duration = Some(0);
                }

                if max_context.is_none() {
                    max_context = Some(-1);
                }

                if max_len.is_none() {
                    max_len = Some(0);
                }

                if split_on_word.is_none() {
                    split_on_word = Some(false);
                }

                Ok(TranslationRequest {
                    file,
                    model,
                    prompt,
                    response_format,
                    temperature,
                    language,
                    detect_language,
                    offset_time,
                    duration,
                    max_context,
                    max_len,
                    split_on_word,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "file",
            "model",
            "prompt",
            "response_format",
            "temperature",
            "language",
            "detect_language",
            "offset_time",
            "duration",
            "max_context",
            "max_len",
            "split_on_word",
        ];
        deserializer.deserialize_struct("TranslationRequest", FIELDS, TranslationRequestVisitor)
    }
}

#[test]
fn test_deserialize_translation_request() {
    let json = r#"{"file":{"id":"file-abc123","bytes":1000,"created_at":1717027200,"filename":"audio.mp3","object":"file","purpose":"assistants"},"model":"whisper-1","language":"en","prompt":"","response_format":"json","temperature":0.0,"detect_language":true,"offset_time":0,"duration":0,"max_context":-1,"max_len":0,"split_on_word":false}"#;
    let obj: TranslationRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(obj.detect_language, Some(true));
    assert_eq!(obj.language, Some("auto".to_string()));
    assert!(obj.prompt.is_none());
}

/// Represents a translation object.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranslationObject {
    /// The translated text.
    pub text: String,
}
