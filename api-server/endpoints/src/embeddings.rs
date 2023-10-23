use crate::common::Usage;
use serde::{Deserialize, Serialize};

pub struct EmbeddingsRequestBuilder {
    req: EmbeddingsRequest,
}
impl EmbeddingsRequestBuilder {
    pub fn new(model: impl Into<String>, input: Vec<String>) -> Self {
        Self {
            req: EmbeddingsRequest {
                model: model.into(),
                input,
                user: None,
            },
        }
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.req.user = Some(user.into());
        self
    }

    pub fn build(self) -> EmbeddingsRequest {
        self.req
    }
}

/// Creates an embedding vector representing the input text.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    /// ID of the model to use.
    model: String,
    /// Input text to embed. Each input must not exceed the max input tokens for the model (8191 tokens for text-embedding-ada-002) and cannot be an empty string.
    input: Vec<String>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: Option<String>,
    pub data: Option<Vec<EmbeddingData>>,
    pub model: String,
    pub usage: Usage,
}

/// Represents an embedding vector returned by embedding endpoint.
#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingData {
    /// The index of the embedding in the list of embeddings.
    index: u32,
    /// The object type, which is always "embedding".
    object: String,
    /// The embedding vector, which is a list of floats.
    embedding: Vec<f64>,
}
