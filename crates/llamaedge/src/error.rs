//! Error types for the LlamaEdge SDK.

use thiserror::Error;

/// Error type for LlamaEdge SDK operations.
#[derive(Debug, Error)]
pub enum Error {
    /// HTTP request failed.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// API returned an error response.
    #[error("API error (status {status}): {message}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Error message from the API.
        message: String,
    },

    /// Invalid URL provided.
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    /// Request timeout.
    #[error("Request timeout")]
    Timeout,

    /// Connection failed.
    #[error("Connection failed: {0}")]
    Connection(String),

    /// Stream error.
    #[error("Stream error: {0}")]
    Stream(String),
}

/// A specialized Result type for LlamaEdge SDK operations.
pub type Result<T> = std::result::Result<T, Error>;
