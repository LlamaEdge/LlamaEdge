//! HTTP client for LlamaEdge API server.

use crate::error::{Error, Result};
use endpoints::{
    audio::{
        speech::{SpeechRequest, SpeechVoice},
        transcription::TranscriptionObject,
        translation::TranslationObject,
    },
    chat::{
        ChatCompletionChunk, ChatCompletionObject, ChatCompletionRequest,
        ChatCompletionRequestBuilder,
    },
    embeddings::{ChunksRequest, ChunksResponse, EmbeddingRequest, EmbeddingsResponse, InputText},
    files::{DeleteFileStatus, FileObject, ListFilesResponse},
    images::{ImageCreateRequest, ListImagesResponse},
    models::ListModelsResponse,
};
use eventsource_stream::Eventsource;
use futures::stream::{Stream, StreamExt};
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE},
    multipart::{Form, Part},
};
use std::pin::Pin;

#[allow(unused_imports)]
use std::time::Duration;

/// Type alias for a boxed stream of chat completion chunks.
pub type ChatCompletionStream =
    Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send + 'static>>;

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

    // ========== Streaming Chat API ==========

    /// Sends a simple chat message and returns a stream of response chunks.
    ///
    /// This is a convenience method for simple streaming conversations.
    ///
    /// # Arguments
    ///
    /// * `message` - The user message to send.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    /// use futures::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let mut stream = client.chat_stream("Tell me a story").await?;
    ///
    ///     while let Some(chunk) = stream.next().await {
    ///         let chunk = chunk?;
    ///         if let Some(choice) = chunk.choices.first() {
    ///             if let Some(ref content) = choice.delta.content {
    ///                 print!("{}", content);
    ///             }
    ///         }
    ///     }
    ///     println!();
    ///     Ok(())
    /// }
    /// ```
    pub async fn chat_stream(&self, message: &str) -> Result<ChatCompletionStream> {
        use endpoints::chat::{ChatCompletionRequestMessage, ChatCompletionUserMessageContent};

        let user_message = ChatCompletionRequestMessage::new_user_message(
            ChatCompletionUserMessageContent::Text(message.to_string()),
            None,
        );

        let request = ChatCompletionRequestBuilder::new(&[user_message])
            .enable_stream(true)
            .build();

        self.chat_completions_stream(&request).await
    }

    /// Sends a streaming chat completion request and returns a stream of chunks.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat completion request. Note: `stream` will be automatically enabled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    /// use endpoints::chat::{ChatCompletionRequestBuilder, ChatCompletionRequestMessage, ChatCompletionUserMessageContent};
    /// use futures::StreamExt;
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
    ///         .enable_stream(true)
    ///         .build();
    ///
    ///     let mut stream = client.chat_completions_stream(&request).await?;
    ///     while let Some(chunk) = stream.next().await {
    ///         let chunk = chunk?;
    ///         if let Some(choice) = chunk.choices.first() {
    ///             if let Some(ref content) = choice.delta.content {
    ///                 print!("{}", content);
    ///             }
    ///         }
    ///     }
    ///     println!();
    ///     Ok(())
    /// }
    /// ```
    pub async fn chat_completions_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(self.build_headers())
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::Api {
                status: status_code,
                message,
            });
        }

        let stream = response
            .bytes_stream()
            .eventsource()
            .filter_map(|event| async {
                match event {
                    Ok(event) => {
                        // Skip [DONE] message
                        if event.data == "[DONE]" {
                            return None;
                        }
                        // Parse the JSON data
                        match serde_json::from_str::<ChatCompletionChunk>(&event.data) {
                            Ok(chunk) => Some(Ok(chunk)),
                            Err(e) => Some(Err(Error::Json(e))),
                        }
                    }
                    Err(e) => Some(Err(Error::Stream(e.to_string()))),
                }
            });

        Ok(Box::pin(stream))
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

    // ========== Audio API ==========

    /// Transcribes audio into the input language.
    ///
    /// # Arguments
    ///
    /// * `audio_data` - The audio file data in bytes.
    /// * `filename` - The filename of the audio file (e.g., "audio.mp3").
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let audio_data = std::fs::read("audio.mp3")?;
    ///     let text = client.transcribe(&audio_data, "audio.mp3").await?;
    ///     println!("Transcription: {}", text);
    ///     Ok(())
    /// }
    /// ```
    pub async fn transcribe(&self, audio_data: &[u8], filename: &str) -> Result<String> {
        let response = self.transcribe_request(audio_data, filename, None).await?;
        Ok(response.text)
    }

    /// Transcribes audio with optional language specification.
    ///
    /// # Arguments
    ///
    /// * `audio_data` - The audio file data in bytes.
    /// * `filename` - The filename of the audio file.
    /// * `language` - Optional ISO-639-1 language code (e.g., "en", "zh").
    pub async fn transcribe_request(
        &self,
        audio_data: &[u8],
        filename: &str,
        language: Option<&str>,
    ) -> Result<TranscriptionObject> {
        let url = format!("{}/v1/audio/transcriptions", self.base_url);

        let file_part = Part::bytes(audio_data.to_vec())
            .file_name(filename.to_string())
            .mime_str("audio/mpeg")
            .map_err(|e| Error::Stream(e.to_string()))?;

        let mut form = Form::new().part("file", file_part);

        if let Some(lang) = language {
            form = form.text("language", lang.to_string());
        }

        let mut request = self.http_client.post(&url);

        if let Some(ref api_key) = self.api_key {
            request = request.header(AUTHORIZATION, format!("Bearer {}", api_key));
        }

        let response = request.multipart(form).send().await?;

        self.handle_response(response).await
    }

    /// Translates audio into English.
    ///
    /// # Arguments
    ///
    /// * `audio_data` - The audio file data in bytes.
    /// * `filename` - The filename of the audio file.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let audio_data = std::fs::read("chinese_audio.mp3")?;
    ///     let english_text = client.translate_audio(&audio_data, "chinese_audio.mp3").await?;
    ///     println!("Translation: {}", english_text);
    ///     Ok(())
    /// }
    /// ```
    pub async fn translate_audio(&self, audio_data: &[u8], filename: &str) -> Result<String> {
        let response = self.translate_audio_request(audio_data, filename).await?;
        Ok(response.text)
    }

    /// Translates audio into English and returns the full response.
    pub async fn translate_audio_request(
        &self,
        audio_data: &[u8],
        filename: &str,
    ) -> Result<TranslationObject> {
        let url = format!("{}/v1/audio/translations", self.base_url);

        let file_part = Part::bytes(audio_data.to_vec())
            .file_name(filename.to_string())
            .mime_str("audio/mpeg")
            .map_err(|e| Error::Stream(e.to_string()))?;

        let form = Form::new().part("file", file_part);

        let mut request = self.http_client.post(&url);

        if let Some(ref api_key) = self.api_key {
            request = request.header(AUTHORIZATION, format!("Bearer {}", api_key));
        }

        let response = request.multipart(form).send().await?;

        self.handle_response(response).await
    }

    /// Generates audio from the input text (text-to-speech).
    ///
    /// # Arguments
    ///
    /// * `text` - The text to generate audio for.
    /// * `model` - The model to use for speech generation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let audio_data = client.speech("Hello, world!", "tts-model").await?;
    ///     std::fs::write("output.wav", audio_data)?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn speech(&self, text: &str, model: &str) -> Result<Vec<u8>> {
        self.speech_with_voice(text, model, None).await
    }

    /// Generates audio from text with a specific voice.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to generate audio for.
    /// * `model` - The model to use for speech generation.
    /// * `voice` - Optional voice to use (alloy, echo, fable, onyx, nova, shimmer).
    pub async fn speech_with_voice(
        &self,
        text: &str,
        model: &str,
        voice: Option<SpeechVoice>,
    ) -> Result<Vec<u8>> {
        let url = format!("{}/v1/audio/speech", self.base_url);

        let request = SpeechRequest {
            model: model.to_string(),
            input: text.to_string(),
            voice,
            response_format: None,
            speed: None,
            speaker_id: None,
            noise_scale: None,
            length_scale: None,
            noise_w: None,
            sentence_silence: None,
            phoneme_silence: None,
            json_input: None,
        };

        let response = self
            .http_client
            .post(&url)
            .headers(self.build_headers())
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::Api {
                status: status_code,
                message,
            });
        }

        let bytes = response.bytes().await?;
        Ok(bytes.to_vec())
    }

    // ========== Images API ==========

    /// Generates an image from a text prompt.
    ///
    /// # Arguments
    ///
    /// * `request` - The image creation request.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    /// use endpoints::images::ImageCreateRequestBuilder;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let request = ImageCreateRequestBuilder::new("sd-model", "A beautiful sunset")
    ///         .with_image_size(512, 512)
    ///         .build();
    ///     let response = client.create_image(&request).await?;
    ///     println!("Generated {} images", response.data.len());
    ///     Ok(())
    /// }
    /// ```
    pub async fn create_image(&self, request: &ImageCreateRequest) -> Result<ListImagesResponse> {
        let url = format!("{}/v1/images/generations", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(self.build_headers())
            .json(request)
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Generates an image with a simple prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text prompt describing the image.
    /// * `model` - The model to use for image generation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let response = client.generate_image("A cat sitting on a windowsill", "sd-model").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn generate_image(&self, prompt: &str, model: &str) -> Result<ListImagesResponse> {
        use endpoints::images::ImageCreateRequestBuilder;
        let request = ImageCreateRequestBuilder::new(model, prompt).build();
        self.create_image(&request).await
    }

    // ========== Files API ==========

    /// Lists all uploaded files.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let files = client.list_files().await?;
    ///     for file in files.data {
    ///         println!("File: {} ({})", file.filename, file.id);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn list_files(&self) -> Result<ListFilesResponse> {
        let url = format!("{}/v1/files", self.base_url);

        let response = self
            .http_client
            .get(&url)
            .headers(self.build_headers())
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Uploads a file to the server.
    ///
    /// # Arguments
    ///
    /// * `file_data` - The file content in bytes.
    /// * `filename` - The name of the file.
    /// * `purpose` - The intended purpose (e.g., "assistants", "fine-tune").
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let file_data = std::fs::read("document.txt")?;
    ///     let file = client.upload_file(&file_data, "document.txt", "assistants").await?;
    ///     println!("Uploaded file ID: {}", file.id);
    ///     Ok(())
    /// }
    /// ```
    pub async fn upload_file(
        &self,
        file_data: &[u8],
        filename: &str,
        purpose: &str,
    ) -> Result<FileObject> {
        let url = format!("{}/v1/files", self.base_url);

        let file_part = Part::bytes(file_data.to_vec())
            .file_name(filename.to_string())
            .mime_str("application/octet-stream")
            .map_err(|e| Error::Stream(e.to_string()))?;

        let form = Form::new()
            .part("file", file_part)
            .text("purpose", purpose.to_string());

        let mut request = self.http_client.post(&url);

        if let Some(ref api_key) = self.api_key {
            request = request.header(AUTHORIZATION, format!("Bearer {}", api_key));
        }

        let response = request.multipart(form).send().await?;

        self.handle_response(response).await
    }

    /// Retrieves information about a specific file.
    ///
    /// # Arguments
    ///
    /// * `file_id` - The ID of the file to retrieve.
    pub async fn get_file(&self, file_id: &str) -> Result<FileObject> {
        let url = format!("{}/v1/files/{}", self.base_url, file_id);

        let response = self
            .http_client
            .get(&url)
            .headers(self.build_headers())
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Deletes a file from the server.
    ///
    /// # Arguments
    ///
    /// * `file_id` - The ID of the file to delete.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let status = client.delete_file("file-abc123").await?;
    ///     println!("Deleted: {}", status.deleted);
    ///     Ok(())
    /// }
    /// ```
    pub async fn delete_file(&self, file_id: &str) -> Result<DeleteFileStatus> {
        let url = format!("{}/v1/files/{}", self.base_url, file_id);

        let response = self
            .http_client
            .delete(&url)
            .headers(self.build_headers())
            .send()
            .await?;

        self.handle_response(response).await
    }

    // ========== Chunks API ==========

    /// Splits a file into text chunks for RAG applications.
    ///
    /// # Arguments
    ///
    /// * `file_id` - The ID of the file to chunk.
    /// * `filename` - The filename.
    /// * `chunk_capacity` - The maximum size of each chunk.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llamaedge::Client;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Client::new("http://localhost:8080");
    ///     let response = client.create_chunks("file-abc123", "document.txt", 512).await?;
    ///     println!("Created {} chunks", response.chunks.len());
    ///     Ok(())
    /// }
    /// ```
    pub async fn create_chunks(
        &self,
        file_id: &str,
        filename: &str,
        chunk_capacity: usize,
    ) -> Result<ChunksResponse> {
        let url = format!("{}/v1/chunks", self.base_url);

        let request = ChunksRequest {
            id: file_id.to_string(),
            filename: filename.to_string(),
            chunk_capacity,
        };

        let response = self
            .http_client
            .post(&url)
            .headers(self.build_headers())
            .json(&request)
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
