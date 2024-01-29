use serde::{Deserialize, Serialize};

/// Creates an embedding vector representing the input text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// ID of the model to use.
    pub model: String,
    /// Input text to embed. Each input must not exceed the max input tokens for the model (8191 tokens for text-embedding-ada-002) and cannot be an empty string.
    pub input: Vec<String>,
    /// The format to return the embeddings in. Can be either float or base64.
    /// Defaults to float.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Represents an embedding vector returned by embedding endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingObject {
    /// The index of the embedding in the list of embeddings.
    pub index: u32,
    /// The object type, which is always "embedding".
    pub object: String,
    /// The embedding vector, which is a list of floats.
    pub embedding: Vec<f64>,
}
