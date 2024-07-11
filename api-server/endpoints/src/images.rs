//! Define types for image generation.

use crate::files::FileObject;
use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use std::fmt;

/// Builder for creating a `ImageCreateRequest` instance.
pub struct ImageCreateRequestBuilder {
    req: ImageCreateRequest,
}
impl ImageCreateRequestBuilder {
    /// Create a new builder with the given model and prompt.
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            req: ImageCreateRequest {
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
    pub fn build(self) -> ImageCreateRequest {
        self.req
    }
}

/// Request to create an image by a given prompt.
#[derive(Debug, Serialize, Default)]
pub struct ImageCreateRequest {
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
impl<'de> Deserialize<'de> for ImageCreateRequest {
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
            type Value = ImageCreateRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CreateImageRequest")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<ImageCreateRequest, V::Error>
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

                Ok(ImageCreateRequest {
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

            fn visit_map<V>(self, mut map: V) -> Result<ImageCreateRequest, V::Error>
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
                Ok(ImageCreateRequest {
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

        const FIELDS: &[&str] = &[
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
fn test_serialize_image_create_request() {
    {
        let req = ImageCreateRequestBuilder::new("test-model-name", "This is a prompt").build();
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"prompt":"This is a prompt","model":"test-model-name","n":1,"response_format":"b64_json"}"#
        );
    }

    {
        let req = ImageCreateRequestBuilder::new("test-model-name", "This is a prompt")
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
fn test_deserialize_image_create_request() {
    {
        let json = r#"{"prompt":"This is a prompt","model":"test-model-name"}"#;
        let req: ImageCreateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "This is a prompt");
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(1));
        assert_eq!(req.response_format, Some(ResponseFormat::B64Json));
    }

    {
        let json = r#"{"prompt":"This is a prompt","model":"test-model-name","n":2,"response_format":"url","size":"1024x1024","style":"vivid","user":"user"}"#;
        let req: ImageCreateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "This is a prompt");
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(2));
        assert_eq!(req.response_format, Some(ResponseFormat::Url));
        assert_eq!(req.size, Some("1024x1024".to_string()));
        assert_eq!(req.style, Some("vivid".to_string()));
        assert_eq!(req.user, Some("user".to_string()));
    }
}

/// Builder for creating a `ImageEditRequest` instance.
pub struct ImageEditRequestBuilder {
    req: ImageEditRequest,
}
impl ImageEditRequestBuilder {
    /// Create a new builder with the given image, prompt, and mask.
    pub fn new(model: impl Into<String>, image: FileObject, prompt: impl Into<String>) -> Self {
        Self {
            req: ImageEditRequest {
                image,
                prompt: prompt.into(),
                mask: None,
                model: model.into(),
                n: Some(1),
                response_format: Some(ResponseFormat::B64Json),
                ..Default::default()
            },
        }
    }

    /// Set an additional image whose fully transparent areas (e.g. where alpha is zero) indicate where `image` should be edited. Must have the same dimensions as `image`.
    pub fn with_mask(mut self, mask: FileObject) -> Self {
        self.req.mask = Some(mask);
        self
    }

    /// Set the number of images to generate.
    pub fn with_number_of_images(mut self, n: u64) -> Self {
        self.req.n = Some(n);
        self
    }

    /// Set the size of the generated images.
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.req.size = Some(size.into());
        self
    }

    /// Set the format in which the generated images are returned.
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.req.response_format = Some(response_format);
        self
    }

    /// Set the user id
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.req.user = Some(user.into());
        self
    }

    /// Build the request.
    pub fn build(self) -> ImageEditRequest {
        self.req
    }
}

/// Request to create an edited or extended image given an original image and a prompt.
#[derive(Debug, Serialize, Default)]
pub struct ImageEditRequest {
    /// The image to edit. If mask is not provided, image must have transparency, which will be used as the mask.
    pub image: FileObject,
    /// A text description of the desired image(s).
    pub prompt: String,
    /// An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where `image` should be edited. Must have the same dimensions as `image`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<FileObject>,
    /// The model to use for image generation.
    pub model: String,
    /// The number of images to generate. Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    /// The size of the generated images. Defaults to 1024x1024.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// The format in which the generated images are returned. Must be one of `url` or `b64_json`. Defaults to `b64_json`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// A unique identifier representing your end-user, which can help monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}
impl<'de> Deserialize<'de> for ImageEditRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Image,
            Prompt,
            Mask,
            Model,
            N,
            Size,
            ResponseFormat,
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
                            "image" => Ok(Field::Image),
                            "prompt" => Ok(Field::Prompt),
                            "mask" => Ok(Field::Mask),
                            "model" => Ok(Field::Model),
                            "n" => Ok(Field::N),
                            "size" => Ok(Field::Size),
                            "response_format" => Ok(Field::ResponseFormat),
                            "user" => Ok(Field::User),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct ImageEditRequestVisitor;

        impl<'de> Visitor<'de> for ImageEditRequestVisitor {
            type Value = ImageEditRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ImageEditRequest")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ImageEditRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut image = None;
                let mut prompt = None;
                let mut mask = None;
                let mut model = None;
                let mut n = None;
                let mut size = None;
                let mut response_format = None;
                let mut user = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Image => {
                            if image.is_some() {
                                return Err(de::Error::duplicate_field("image"));
                            }
                            image = Some(map.next_value()?);
                        }
                        Field::Prompt => {
                            if prompt.is_some() {
                                return Err(de::Error::duplicate_field("prompt"));
                            }
                            prompt = Some(map.next_value()?);
                        }
                        Field::Mask => {
                            if mask.is_some() {
                                return Err(de::Error::duplicate_field("mask"));
                            }
                            mask = Some(map.next_value()?);
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
                        Field::Size => {
                            if size.is_some() {
                                return Err(de::Error::duplicate_field("size"));
                            }
                            size = Some(map.next_value()?);
                        }
                        Field::ResponseFormat => {
                            if response_format.is_some() {
                                return Err(de::Error::duplicate_field("response_format"));
                            }
                            response_format = Some(map.next_value()?);
                        }
                        Field::User => {
                            if user.is_some() {
                                return Err(de::Error::duplicate_field("user"));
                            }
                            user = Some(map.next_value()?);
                        }
                    }
                }
                Ok(ImageEditRequest {
                    image: image.ok_or_else(|| de::Error::missing_field("image"))?,
                    prompt: prompt.ok_or_else(|| de::Error::missing_field("prompt"))?,
                    mask,
                    model: model.ok_or_else(|| de::Error::missing_field("model"))?,
                    n: n.unwrap_or(Some(1)),
                    size,
                    response_format: response_format.unwrap_or(Some(ResponseFormat::B64Json)),
                    user,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "image",
            "prompt",
            "mask",
            "model",
            "n",
            "size",
            "response_format",
            "user",
        ];
        deserializer.deserialize_struct("ImageEditRequest", FIELDS, ImageEditRequestVisitor)
    }
}

#[test]
fn test_serialize_image_edit_request() {
    {
        let req = ImageEditRequestBuilder::new(
            "test-model-name",
            FileObject {
                id: "test-image-id".to_string(),
                bytes: 1024,
                created_at: 1234567890,
                filename: "test-image.png".to_string(),
                object: "file".to_string(),
                purpose: "fine-tune".to_string(),
            },
            "This is a prompt",
        )
        .build();
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","model":"test-model-name","n":1,"response_format":"b64_json"}"#
        );
    }

    {
        let req = ImageEditRequestBuilder::new(
            "test-model-name",
            FileObject {
                id: "test-image-id".to_string(),
                bytes: 1024,
                created_at: 1234567890,
                filename: "test-image.png".to_string(),
                object: "file".to_string(),
                purpose: "fine-tune".to_string(),
            },
            "This is a prompt",
        )
        .with_number_of_images(2)
        .with_response_format(ResponseFormat::Url)
        .with_size("256x256")
        .with_user("user")
        .build();
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","model":"test-model-name","n":2,"size":"256x256","response_format":"url","user":"user"}"#
        );
    }
}

#[test]
fn test_deserialize_image_edit_request() {
    {
        let json = r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","model":"test-model-name"}"#;
        let req: ImageEditRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.image.id, "test-image-id");
        assert_eq!(req.image.bytes, 1024);
        assert_eq!(req.image.created_at, 1234567890);
        assert_eq!(req.image.filename, "test-image.png");
        assert_eq!(req.image.object, "file");
        assert_eq!(req.image.purpose, "fine-tune");
        assert_eq!(req.prompt, "This is a prompt");
        assert!(req.mask.is_none());
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(1));
        assert_eq!(req.response_format, Some(ResponseFormat::B64Json));
    }

    {
        let json = r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","model":"test-model-name","n":2,"size":"256x256","response_format":"url","user":"user"}"#;
        let req: ImageEditRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.image.id, "test-image-id");
        assert_eq!(req.image.bytes, 1024);
        assert_eq!(req.image.created_at, 1234567890);
        assert_eq!(req.image.filename, "test-image.png");
        assert_eq!(req.image.object, "file");
        assert_eq!(req.image.purpose, "fine-tune");
        assert_eq!(req.prompt, "This is a prompt");
        assert!(req.mask.is_none());
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(2));
        assert_eq!(req.size, Some("256x256".to_string()));
        assert_eq!(req.response_format, Some(ResponseFormat::Url));
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

/// Represents the url or the content of an image generated.
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct ImageObject {
    /// The base64-encoded JSON of the generated image, if response_format is `b64_json`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    /// The URL of the generated image, if response_format is `url`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The prompt that was used to generate the image, if there was any revision to the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}
