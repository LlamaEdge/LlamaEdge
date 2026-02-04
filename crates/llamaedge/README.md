# LlamaEdge Rust SDK

A Rust SDK for interacting with [llama-api-server](https://github.com/LlamaEdge/LlamaEdge), which provides OpenAI-compatible REST APIs for local LLM inference.

## Features

- Simple and ergonomic API
- Support for chat completions (streaming and non-streaming)
- Support for embeddings generation
- Support for listing available models
- Configurable timeout and API key authentication

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
llamaedge = "0.1"
```

## Quick Start

```rust
use llamaedge::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a client
    let client = Client::new("http://localhost:8080");

    // Simple chat
    let response = client.chat("What is Rust?").await?;
    println!("{}", response);

    Ok(())
}
```

## Streaming Responses

```rust
use llamaedge::{Client, StreamExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://localhost:8080");

    // Stream chat responses
    let mut stream = client.chat_stream("Tell me a story").await?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(choice) = chunk.choices.first() {
            if let Some(ref content) = choice.delta.content {
                print!("{}", content);
            }
        }
    }

    Ok(())
}
```

## Multi-turn Conversation

```rust
use llamaedge::Client;
use endpoints::chat::{
    ChatCompletionRequestBuilder,
    ChatCompletionRequestMessage,
    ChatCompletionUserMessageContent,
    ChatCompletionSystemMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://localhost:8080");

    // Create messages
    let system_message = ChatCompletionRequestMessage::System(
        ChatCompletionSystemMessage::new("You are a helpful assistant.", None),
    );
    let user_message = ChatCompletionRequestMessage::new_user_message(
        ChatCompletionUserMessageContent::Text("Hello!".to_string()),
        None,
    );

    // Build request
    let request = ChatCompletionRequestBuilder::new(&[system_message, user_message])
        .with_model("llama3")
        .build();

    // Send request
    let response = client.chat_completions(&request).await?;
    if let Some(choice) = response.choices.first() {
        if let Some(ref content) = choice.message.content {
            println!("{}", content);
        }
    }

    Ok(())
}
```

## Embeddings

```rust
use llamaedge::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://localhost:8080");

    // Generate embedding for a single text
    let embedding = client.embeddings("Hello, world!").await?;
    println!("Embedding dimension: {}", embedding.len());

    // Generate embeddings for multiple texts
    let embeddings = client.embeddings_batch(&["Hello", "World"]).await?;
    for (i, emb) in embeddings.iter().enumerate() {
        println!("Embedding {}: {} dimensions", i, emb.len());
    }

    Ok(())
}
```

## Development (within monorepo)

When developing this SDK within the LlamaEdge monorepo, you need to specify the target explicitly because the workspace is configured for WASM targets:

```bash
# Using Make (recommended)
make build
make test
make check

# Or using cargo directly
cargo build --target aarch64-apple-darwin  # macOS ARM64
cargo build --target x86_64-unknown-linux-gnu  # Linux
cargo test --target aarch64-apple-darwin
```

## License

Apache-2.0