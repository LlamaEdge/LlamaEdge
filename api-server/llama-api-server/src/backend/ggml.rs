use crate::error;
use chat_prompts::PromptTemplateType;
use endpoints::{
    chat::{ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionUserMessageContent},
    completions::CompletionRequest,
    embeddings::EmbeddingRequest,
    rag::{RagChatCompletionsRequest, RagEmbeddingRequest},
};
use futures_util::TryStreamExt;
use hyper::{body::to_bytes, Body, Request, Response};

/// Lists models available
pub(crate) async fn models_handler() -> Result<Response<Body>, hyper::Error> {
    let list_models_response = match llama_core::models::models().await {
        Ok(list_models_response) => list_models_response,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    // serialize response
    let s = match serde_json::to_string(&list_models_response) {
        Ok(s) => s,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .body(Body::from(s));
    match result {
        Ok(response) => Ok(response),
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

pub(crate) async fn embeddings_handler(
    mut req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let embedding_request: EmbeddingRequest = match serde_json::from_slice(&body_bytes) {
        Ok(embedding_request) => embedding_request,
        Err(e) => {
            return error::bad_request(format!(
                "Fail to parse embedding request: {msg}",
                msg = e.to_string()
            ));
        }
    };

    match llama_core::embeddings::embeddings(&embedding_request).await {
        Ok(embedding_response) => {
            // serialize embedding object
            match serde_json::to_string(&embedding_response) {
                Ok(s) => {
                    // return response
                    let result = Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .body(Body::from(s));
                    match result {
                        Ok(response) => Ok(response),
                        Err(e) => error::internal_server_error(e.to_string()),
                    }
                }
                Err(e) => error::internal_server_error(format!(
                    "Fail to serialize embedding object. {}",
                    e.to_string()
                )),
            }
        }
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

pub(crate) async fn completions_handler(
    mut req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let completion_request: CompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(completion_request) => completion_request,
        Err(e) => {
            return error::bad_request(format!(
                "Failed to deserialize completion request. {msg}",
                msg = e.to_string()
            ));
        }
    };

    match llama_core::completions::completions(&completion_request).await {
        Ok(completion_object) => {
            // serialize completion object
            let s = match serde_json::to_string(&completion_object) {
                Ok(s) => s,
                Err(e) => {
                    return error::internal_server_error(format!(
                        "Failed to serialize completion object. {msg}",
                        msg = e.to_string()
                    ));
                }
            };

            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .body(Body::from(s));
            match result {
                Ok(response) => Ok(response),
                Err(e) => error::internal_server_error(e.to_string()),
            }
        }
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

/// Processes a chat-completion request and returns a chat-completion response with the answer from the model.
pub(crate) async fn chat_completions_handler(
    mut req: Request<Body>,
    template_ty: PromptTemplateType,
    log_prompts: bool,
) -> Result<Response<Body>, hyper::Error> {
    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .body(Body::empty());

        match result {
            Ok(response) => return Ok(response),
            Err(e) => {
                return error::internal_server_error(e.to_string());
            }
        }
    }

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let chat_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chat_request) => chat_request,
        Err(e) => {
            return error::bad_request(format!(
                "Fail to parse chat completion request: {msg}",
                msg = e.to_string()
            ));
        }
    };

    match chat_request.stream {
        Some(true) => chat_completions_stream(chat_request, template_ty, log_prompts).await,
        Some(false) | None => chat_completions(chat_request, template_ty, log_prompts).await,
    }
}

/// Processes a chat-completion request in stream mode and returns a chat-completion response with the answer from the model.
async fn chat_completions_stream(
    mut chat_request: ChatCompletionRequest,
    template_ty: PromptTemplateType,
    log_prompts: bool,
) -> Result<Response<Body>, hyper::Error> {
    match llama_core::chat::chat_completions_stream(&mut chat_request, template_ty, log_prompts)
        .await
    {
        Ok(stream) => {
            let stream = stream.map_err(|e| e.to_string());

            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .body(Body::wrap_stream(stream));

            match result {
                Ok(response) => Ok(response),
                Err(e) => error::internal_server_error(e.to_string()),
            }
        }
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

/// Processes a chat-completion request and returns a chat-completion response with the answer from the model.
async fn chat_completions(
    mut chat_request: ChatCompletionRequest,
    template_ty: PromptTemplateType,
    log_prompts: bool,
) -> Result<Response<Body>, hyper::Error> {
    match llama_core::chat::chat_completions(&mut chat_request, template_ty, log_prompts).await {
        Ok(chat_completion_object) => {
            // serialize chat completion object
            let s = match serde_json::to_string(&chat_completion_object) {
                Ok(s) => s,
                Err(e) => {
                    return error::internal_server_error(format!(
                        "Fail to serialize chat completion object. {}",
                        e.to_string()
                    ));
                }
            };

            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .body(Body::from(s));

            match result {
                Ok(response) => Ok(response),
                Err(e) => error::internal_server_error(e.to_string()),
            }
        }
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

pub(crate) async fn rag_doc_chunks_to_embeddings_handler(
    mut req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let rag_embedding_request: RagEmbeddingRequest = match serde_json::from_slice(&body_bytes) {
        Ok(embedding_request) => embedding_request,
        Err(e) => {
            return error::bad_request(format!(
                "Fail to parse embedding request: {msg}",
                msg = e.to_string()
            ));
        }
    };

    match llama_core::rag::rag_doc_chunks_to_embeddings(&rag_embedding_request).await {
        Ok(collection_name) => {
            let s = format!("The embeddings for the document chunks has been computed and persisted in the Qdrant collection '{}'.", collection_name);

            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .body(Body::from(s));
            match result {
                Ok(response) => Ok(response),
                Err(e) => error::internal_server_error(e.to_string()),
            }
        }
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

pub(crate) async fn rag_query_handler(
    mut req: Request<Body>,
    template_ty: PromptTemplateType,
    log_prompts: bool,
) -> Result<Response<Body>, hyper::Error> {
    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .body(Body::empty());

        match result {
            Ok(response) => return Ok(response),
            Err(e) => {
                return error::internal_server_error(e.to_string());
            }
        }
    }

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let rag_chat_request: RagChatCompletionsRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chat_request) => chat_request,
        Err(e) => {
            return error::bad_request(format!(
                "Fail to parse chat completion request: {msg}",
                msg = e.to_string()
            ));
        }
    };

    // compute embeddings for user query
    let embedding_response = match rag_chat_request.messages.is_empty() {
        true => return error::bad_request("Messages cannot be empty"),
        false => {
            let last_message = rag_chat_request.messages.last().unwrap();
            match last_message {
                ChatCompletionRequestMessage::User(user_message) => {
                    let query_text = match user_message.content() {
                        ChatCompletionUserMessageContent::Text(text) => text,
                        _ => {
                            return error::bad_request(
                                "The last message must be a text content user message",
                            )
                        }
                    };

                    let rag_embedding_request = RagEmbeddingRequest {
                        embedding_request: EmbeddingRequest {
                            model: rag_chat_request.embedding_model.clone(),
                            input: vec![query_text.clone()],
                            encoding_format: rag_chat_request.encoding_format.clone(),
                            user: rag_chat_request.user.clone(),
                        },
                        qdrant_url: rag_chat_request.qdrant_url.clone(),
                        qdrant_collection_name: rag_chat_request.qdrant_collection_name.clone(),
                    };

                    // compute embeddings for query
                    match llama_core::rag::rag_query_to_embeddings(&rag_embedding_request).await {
                        Ok(embedding_response) => embedding_response,
                        Err(e) => {
                            return error::internal_server_error(e.to_string());
                        }
                    }
                }
                _ => return error::bad_request("The last message must be a user message"),
            }
        }
    };
    let query_embedding: Vec<f32> = match embedding_response.data.first() {
        Some(embedding) => embedding.embedding.iter().map(|x| *x as f32).collect(),
        None => return error::internal_server_error("No embeddings returned"),
    };

    // retrieve context
    let scored_points = match llama_core::rag::rag_retrieve_context(
        query_embedding.as_slice(),
        rag_chat_request.qdrant_url.as_str(),
        rag_chat_request.qdrant_collection_name.as_str(),
        rag_chat_request.limit as usize,
    )
    .await
    {
        Ok(search_result) => search_result,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    // prepare chat completion request
    let mut chat_request = rag_chat_request.as_chat_completions_request();

    // update messages with retrieved context
    let mut context = String::new();
    for (idx, point) in scored_points.iter().enumerate() {
        println!("    * Point {}: score: {}", idx, point.score);

        if let Some(payload) = &point.payload {
            if let Some(source) = payload.get("source") {
                println!("      Source: {}", source);

                context.push_str(source.to_string().as_str());
                context.push_str("\n\n");
            }
        }
    }
    println!("\n\n");

    // prepare system message
    let content = format!("Use the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{}", context.trim_end());

    // create system message
    let system_message =
        ChatCompletionRequestMessage::new_system_message(content, rag_chat_request.user.clone());

    // update or insert system message
    match chat_request.messages[0] {
        ChatCompletionRequestMessage::System(_) => {
            chat_request.messages[0] = system_message;
        }
        _ => {
            chat_request.messages.insert(0, system_message);
        }
    }

    // chat completion
    match chat_request.stream {
        Some(true) => chat_completions_stream(chat_request, template_ty, log_prompts).await,
        Some(false) | None => chat_completions(chat_request, template_ty, log_prompts).await,
    }
}
