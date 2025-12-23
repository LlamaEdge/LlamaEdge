//! HTTP client for LlamaEdge API server.

use crate::error::{Error, Result};
use endpoints::{
    chat::{ChatCompletionObject, ChatCompletionRequest, ChatCompletionRequestBuilder},
    embeddings::{EmbeddingRequest, EmbeddingsResponse, InputText},
    models::ListModelsResponse,
};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};

#[allow(unused_imports)]
use std::time::Duration;

/// LlamaEdge API client.
///
/// This client provides methods to interact with the llama-api-server,
/// which exposes OpenAI-compatible REST APIs.
///
/// # Example
///
/// ```no_run
/// use llamaedge::Client;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = Client::new("http://localhost:8080");
///
///     // Simple chat
///     let response = client.chat("What is Rust?").await?;
///     println!("{}", response);
///
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Client {
    /// Base URL of the llama-api-server.
    base_url: String,
    /// HTTP client instance.
    http_client: reqwest::Client,
    /// Optional API key for authentication.
    api_key: Option<String>,
}

impl Client {
    /// Creates a new client with the specified base URL.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL of the llama-api-server (e.g., "http://localhost:8080").
    ///
    /// # Example
    ///
    /// ```
    /// use llamaedge::Client;
    ///
    /// let client = Client::new("http://localhost:8080");
    /// ```
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();

        let http_client = reqwest::Client::builder()
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url,
            http_client,
            api_key: None,
        }
    }

    /// Sets the API key for authentication.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key to use for authentication.
    ///
    /// # Example
    ///
    /// ```
    /// use llamaedge::Client;
    ///
    /// let client = Client::new("http://localhost:8080")
    ///     .with_api_key("your-api-key");
    /// ```
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the request timeout.
    ///
    /// Note: This method only has effect on native targets (not WebAssembly).
    ///
    /// # Arguments
    ///
    /// * `timeout` - The timeout duration for HTTP requests.
    ///
    /// # Example
    ///
    /// ```
    /// use llamaedge::Client;
    /// use std::time::Duration;
    ///
    /// let client = Client::new("http://localhost:8080")
    ///     .with_timeout(Duration::from_secs(60));
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.http_client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");
        self
    }

    /// Sets the request timeout (no-op on WebAssembly targets).
    #[cfg(target_arch = "wasm32")]
    pub fn with_timeout(self, _timeout: Duration) -> Self {
        // Timeout is not supported on wasm32 targets
        self
    }

    /// Returns the base URL of the server.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Builds the default headers for requests.
    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(ref api_key) = self.api_key {
            if let Ok(value) = HeaderValue::from_str(&format!("Bearer {}", api_key)) {
                headers.insert(AUTHORIZATION, value);
            }
        }

        headers
    }

    // ========== Chat API ==========

    /// Sends a simple chat message and returns the response content.
    ///
    /// This is a convenience method for simple single-turn conversations.
    ///
    /// # Arguments
    ///
    /// * `message` - The user message to send.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let response = client.chat("What is Rust?").await?;
    ///     println!("{}", response);
    ///     Ok(())
    /// }
    /// ```
    pub async fn chat(&self, message: &str) -> Result<String> {
        use endpoints::chat::{ChatCompletionRequestMessage, ChatCompletionUserMessageContent};

        let user_message = ChatCompletionRequestMessage::new_user_message(
            ChatCompletionUserMessageContent::Text(message.to_string()),
            None,
        );

        let request = ChatCompletionRequestBuilder::new(&[user_message]).build();

        let response = self.chat_completions(&request).await?;

        // Extract the content from the first choice
        if let Some(choice) = response.choices.first() {
            if let Some(ref content) = choice.message.content {
                return Ok(content.clone());
            }
        }

        Ok(String::new())
    }

    /// Sends a chat completion request and returns the full response.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat completion request.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    /// use endpoints::chat::{ChatCompletionRequestBuilder, ChatCompletionRequestMessage, ChatCompletionUserMessageContent};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///
    ///     let user_message = ChatCompletionRequestMessage::new_user_message(
    ///         ChatCompletionUserMessageContent::Text("Hello!".to_string()),
    ///         None,
    ///     );
    ///     let request = ChatCompletionRequestBuilder::new(&[user_message])
    ///         .with_model("llama3")
    ///         .build();
    ///
    ///     let response = client.chat_completions(&request).await?;
    ///     println!("{:?}", response);
    ///     Ok(())
    /// }
    /// ```
    pub async fn chat_completions(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionObject> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(self.build_headers())
            .json(request)
            .send()
            .await?;

        self.handle_response(response).await
    }

    // ========== Embeddings API ==========

    /// Generates an embedding vector for the given input text.
    ///
    /// # Arguments
    ///
    /// * `input` - The text to embed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let embedding = client.embeddings("Hello, world!").await?;
    ///     println!("Embedding dimension: {}", embedding.len());
    ///     Ok(())
    /// }
    /// ```
    pub async fn embeddings(&self, input: &str) -> Result<Vec<f64>> {
        let request = EmbeddingRequest {
            model: None,
            input: InputText::from(input),
            encoding_format: None,
            user: None,
        };

        let response = self.embeddings_request(&request).await?;

        // Extract the first embedding
        if let Some(embedding) = response.data.first() {
            return Ok(embedding.embedding.clone());
        }

        Ok(Vec::new())
    }

    /// Generates embedding vectors for multiple inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The texts to embed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let embeddings = client.embeddings_batch(&["Hello", "World"]).await?;
    ///     for (i, emb) in embeddings.iter().enumerate() {
    ///         println!("Embedding {}: {} dimensions", i, emb.len());
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn embeddings_batch(&self, inputs: &[&str]) -> Result<Vec<Vec<f64>>> {
        let input_strings: Vec<String> = inputs.iter().map(|s| s.to_string()).collect();
        let request = EmbeddingRequest {
            model: None,
            input: InputText::from(input_strings),
            encoding_format: None,
            user: None,
        };

        let response = self.embeddings_request(&request).await?;

        Ok(response.data.into_iter().map(|e| e.embedding).collect())
    }

    /// Sends an embedding request and returns the full response.
    ///
    /// # Arguments
    ///
    /// * `request` - The embedding request.
    pub async fn embeddings_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingsResponse> {
        let url = format!("{}/v1/embeddings", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(self.build_headers())
            .json(request)
            .send()
            .await?;

        self.handle_response(response).await
    }

    // ========== Models API ==========

    /// Lists all available models.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let models = client.list_models().await?;
    ///     for model in models.data {
    ///         println!("Model: {}", model.id);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn list_models(&self) -> Result<ListModelsResponse> {
        let url = format!("{}/v1/models", self.base_url);

        let response = self
            .http_client
            .get(&url)
            .headers(self.build_headers())
            .send()
            .await?;

        self.handle_response(response).await
    }

    // ========== Helper Methods ==========

    /// Handles the HTTP response, parsing JSON or returning an error.
    async fn handle_response<T>(&self, response: reqwest::Response) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let status = response.status();

        if status.is_success() {
            let body = response.json::<T>().await?;
            Ok(body)
        } else {
            let status_code = status.as_u16();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            Err(Error::Api {
                status: status_code,
                message,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = Client::new("http://localhost:8080");
        assert_eq!(client.base_url(), "http://localhost:8080");
    }

    #[test]
    fn test_client_with_trailing_slash() {
        let client = Client::new("http://localhost:8080/");
        assert_eq!(client.base_url(), "http://localhost:8080");
    }

    #[test]
    fn test_client_with_api_key() {
        let client = Client::new("http://localhost:8080").with_api_key("test-key");
        assert!(client.api_key.is_some());
        assert_eq!(client.api_key.as_ref().unwrap(), "test-key");
    }

    #[test]
    fn test_client_with_timeout() {
        let client = Client::new("http://localhost:8080").with_timeout(Duration::from_secs(60));
        // Just verify it doesn't panic
        assert_eq!(client.base_url(), "http://localhost:8080");
    }
}
