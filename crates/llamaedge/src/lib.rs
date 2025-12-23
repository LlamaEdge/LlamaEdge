//! # LlamaEdge Rust SDK
//!
//! A Rust SDK for interacting with [llama-api-server](https://github.com/LlamaEdge/LlamaEdge),
//! which provides OpenAI-compatible REST APIs for local LLM inference.
//!
//! ## Features
//!
//! - Simple and ergonomic API
//! - Support for chat completions (streaming and non-streaming)
//! - Support for embeddings generation
//! - Support for listing available models
//! - Audio API (transcription, translation, text-to-speech)
//! - Image generation API
//! - Files management API
//! - Text chunking for RAG applications
//! - Configurable timeout and API key authentication
//!
//! ## Quick Start
//!
//! ```no_run
//! use llamaedge::Client;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a client
//!     let client = Client::new("http://localhost:8080");
//!
//!     // Simple chat
//!     let response = client.chat("What is Rust?").await?;
//!     println!("{}", response);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Multi-turn Conversation
//!
//! ```no_run
//! use llamaedge::Client;
//! use endpoints::chat::{
//!     ChatCompletionRequestBuilder,
//!     ChatCompletionRequestMessage,
//!     ChatCompletionUserMessageContent,
//!     ChatCompletionSystemMessage,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Client::new("http://localhost:8080");
//!
//!     // Create messages
//!     let system_message = ChatCompletionRequestMessage::System(
//!         ChatCompletionSystemMessage::new("You are a helpful assistant.", None),
//!     );
//!     let user_message = ChatCompletionRequestMessage::new_user_message(
//!         ChatCompletionUserMessageContent::Text("Hello!".to_string()),
//!         None,
//!     );
//!
//!     // Build request
//!     let request = ChatCompletionRequestBuilder::new(&[system_message, user_message])
//!         .with_model("llama3")
//!         .build();
//!
//!     // Send request
//!     let response = client.chat_completions(&request).await?;
//!     if let Some(choice) = response.choices.first() {
//!         if let Some(ref content) = choice.message.content {
//!             println!("{}", content);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Embeddings
//!
//! ```no_run
//! use llamaedge::Client;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Client::new("http://localhost:8080");
//!
//!     // Generate embedding for a single text
//!     let embedding = client.embeddings("Hello, world!").await?;
//!     println!("Embedding dimension: {}", embedding.len());
//!
//!     // Generate embeddings for multiple texts
//!     let embeddings = client.embeddings_batch(&["Hello", "World"]).await?;
//!     for (i, emb) in embeddings.iter().enumerate() {
//!         println!("Embedding {}: {} dimensions", i, emb.len());
//!     }
//!
//!     Ok(())
//! }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

mod client;
mod error;

pub use client::{ChatCompletionStream, Client};
pub use error::{Error, Result};

// Re-export commonly used types from endpoints crate
pub use endpoints::{
    audio::{
        speech::{SpeechRequest, SpeechVoice},
        transcription::TranscriptionObject,
        translation::TranslationObject,
    },
    chat::{
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkChoiceDelta,
        ChatCompletionObject, ChatCompletionRequest, ChatCompletionRequestBuilder,
        ChatCompletionRequestMessage, ChatCompletionSystemMessage,
        ChatCompletionUserMessageContent,
    },
    embeddings::{ChunksRequest, ChunksResponse, EmbeddingRequest, EmbeddingsResponse, InputText},
    files::{DeleteFileStatus, FileObject, ListFilesResponse},
    images::{ImageCreateRequest, ImageCreateRequestBuilder, ListImagesResponse},
    models::{ListModelsResponse, Model},
};

// Re-export futures for streaming
pub use futures::stream::StreamExt;
