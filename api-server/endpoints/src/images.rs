//! Define types for image generation.

use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
// use serde::{de, Deserialize, Deserializer};
use std::fmt;

/// Builder for creating a `CreateImageRequest` instance.
pub struct CreateImageRequestBuilder {
    req: CreateImageRequest,
}
impl CreateImageRequestBuilder {
    /// Create a new builder with the given model and prompt.
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            req: CreateImageRequest {
                model: model.into(),
                prompt: prompt.into(),
                n: Some(1),
                response_format: Some(ResponseFormat::B64Json),
                ..Default::default()
            },
        }
    }

    /// Set the number of images to generate.
    pub fn with_number_of_images(mut self, n: u64) -> Self {
        self.req.n = Some(n);
        self
    }

    /// This param is only supported for OpenAI `dall-e-3`.
    pub fn with_quality(mut self, quality: impl Into<String>) -> Self {
        self.req.quality = Some(quality.into());
        self
    }

    /// Set the format in which the generated images are returned.
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.req.response_format = Some(response_format);
        self
    }

    /// Set the size of the generated images.
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.req.size = Some(size.into());
        self
    }

    /// This param is only supported for `dall-e-3`.
    pub fn with_style(mut self, style: impl Into<String>) -> Self {
        self.req.style = Some(style.into());
        self
    }

    /// Set the user id
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.req.user = Some(user.into());
        self
    }

    /// Build the request.
    pub fn build(self) -> CreateImageRequest {
        self.req
    }
}

/// Request to create an image by a given prompt.
#[derive(Debug, Serialize, Default)]
pub struct CreateImageRequest {
    /// A text description of the desired image.
    pub prompt: String,
    /// Name of the model to use for image generation.
    pub model: String,
    /// Number of images to generate. Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    /// The quality of the image that will be generated. hd creates images with finer details and greater consistency across the image. Defaults to "standard". This param is only supported for OpenAI `dall-e-3`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// The format in which the generated images are returned. Must be one of `url` or `b64_json`. Defaults to `b64_json`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// The size of the generated images. Defaults to 1024x1024.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// The style of the generated images. Must be one of `vivid` or `natural`. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images. This param is only supported for `dall-e-3`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// A unique identifier representing your end-user, which can help monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}
impl<'de> Deserialize<'de> for CreateImageRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Prompt,
            Model,
            N,
            Quality,
            ResponseFormat,
            Size,
            Style,
            User,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("field identifier")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "prompt" => Ok(Field::Prompt),
                            "model" => Ok(Field::Model),
                            "n" => Ok(Field::N),
                            "quality" => Ok(Field::Quality),
                            "response_format" => Ok(Field::ResponseFormat),
                            "size" => Ok(Field::Size),
                            "style" => Ok(Field::Style),
                            "user" => Ok(Field::User),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct CreateImageRequestVisitor;

        impl<'de> Visitor<'de> for CreateImageRequestVisitor {
            type Value = CreateImageRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CreateImageRequest")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<CreateImageRequest, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let prompt = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let model = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let n = seq.next_element()?.unwrap_or(Some(1));
                let quality = seq.next_element()?;
                let response_format = seq.next_element()?.unwrap_or(Some(ResponseFormat::B64Json));
                let size = seq.next_element()?;
                let style = seq.next_element()?;
                let user = seq.next_element()?;

                Ok(CreateImageRequest {
                    prompt,
                    model,
                    n,
                    quality,
                    response_format,
                    size,
                    style,
                    user,
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<CreateImageRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut prompt = None;
                let mut model = None;
                let mut n = None;
                let mut quality = None;
                let mut response_format = None;
                let mut size = None;
                let mut style = None;
                let mut user = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Prompt => {
                            if prompt.is_some() {
                                return Err(de::Error::duplicate_field("prompt"));
                            }
                            prompt = Some(map.next_value()?);
                        }
                        Field::Model => {
                            if model.is_some() {
                                return Err(de::Error::duplicate_field("model"));
                            }
                            model = Some(map.next_value()?);
                        }
                        Field::N => {
                            if n.is_some() {
                                return Err(de::Error::duplicate_field("n"));
                            }
                            n = Some(map.next_value()?);
                        }
                        Field::Quality => {
                            if quality.is_some() {
                                return Err(de::Error::duplicate_field("quality"));
                            }
                            quality = Some(map.next_value()?);
                        }
                        Field::ResponseFormat => {
                            if response_format.is_some() {
                                return Err(de::Error::duplicate_field("response_format"));
                            }
                            response_format = Some(map.next_value()?);
                        }
                        Field::Size => {
                            if size.is_some() {
                                return Err(de::Error::duplicate_field("size"));
                            }
                            size = Some(map.next_value()?);
                        }
                        Field::Style => {
                            if style.is_some() {
                                return Err(de::Error::duplicate_field("style"));
                            }
                            style = Some(map.next_value()?);
                        }
                        Field::User => {
                            if user.is_some() {
                                return Err(de::Error::duplicate_field("user"));
                            }
                            user = Some(map.next_value()?);
                        }
                    }
                }
                Ok(CreateImageRequest {
                    prompt: prompt.ok_or_else(|| de::Error::missing_field("prompt"))?,
                    model: model.ok_or_else(|| de::Error::missing_field("model"))?,
                    n: n.unwrap_or(Some(1)),
                    quality,
                    response_format: response_format.unwrap_or(Some(ResponseFormat::B64Json)),
                    size,
                    style,
                    user,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &[
            "prompt",
            "model",
            "n",
            "quality",
            "response_format",
            "size",
            "style",
            "user",
        ];
        deserializer.deserialize_struct("CreateImageRequest", FIELDS, CreateImageRequestVisitor)
    }
}

#[test]
fn test_serialize_create_image_request() {
    {
        let req = CreateImageRequestBuilder::new("test-model-name", "This is a prompt").build();
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"prompt":"This is a prompt","model":"test-model-name","n":1,"response_format":"b64_json"}"#
        );
    }

    {
        let req = CreateImageRequestBuilder::new("test-model-name", "This is a prompt")
            .with_number_of_images(2)
            .with_response_format(ResponseFormat::Url)
            .with_size("1024x1024")
            .with_style("vivid")
            .with_user("user")
            .build();
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"prompt":"This is a prompt","model":"test-model-name","n":2,"response_format":"url","size":"1024x1024","style":"vivid","user":"user"}"#
        );
    }
}

#[test]
fn test_deserialize_create_image_request() {
    {
        let json = r#"{"prompt":"This is a prompt","model":"test-model-name"}"#;
        let req: CreateImageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "This is a prompt");
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(1));
        assert_eq!(req.response_format, Some(ResponseFormat::B64Json));
    }

    {
        let json = r#"{"prompt":"This is a prompt","model":"test-model-name","n":2,"response_format":"url","size":"1024x1024","style":"vivid","user":"user"}"#;
        let req: CreateImageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "This is a prompt");
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(2));
        assert_eq!(req.response_format, Some(ResponseFormat::Url));
        assert_eq!(req.size, Some("1024x1024".to_string()));
        assert_eq!(req.style, Some("vivid".to_string()));
        assert_eq!(req.user, Some("user".to_string()));
    }
}

/// The format in which the generated images are returned.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    #[serde(rename = "url")]
    Url,
    #[serde(rename = "b64_json")]
    B64Json,
}
