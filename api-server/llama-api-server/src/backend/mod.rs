pub(crate) mod ggml;

use crate::error;
use hyper::{Body, Request, Response};
use llama_core::RunningMode;

pub(crate) async fn handle_llama_request(
    req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    let running_mode = match llama_core::running_mode() {
        Ok(mode) => mode,
        Err(e) => return error::internal_server_error(e.to_string()),
    };

    match running_mode {
        RunningMode::Chat => handle_chat_request(req).await,
        RunningMode::Embeddings => handle_embedding_request(req).await,
        RunningMode::ChatEmbedding => handle_chat_embedding_request(req).await,
        _ => error::not_implemented(),
    }
}

async fn handle_chat_request(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/v1/models" => ggml::models_handler().await,
        "/v1/chat/completions" => ggml::chat_completions_handler(req).await,
        "/v1/completions" => ggml::completions_handler(req).await,
        _ => error::invalid_endpoint(req.uri().path()),
    }
}

async fn handle_embedding_request(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/v1/models" => ggml::models_handler().await,
        "/v1/embeddings" => ggml::embeddings_handler(req).await,
        "/v1/files" => ggml::files_handler(req).await,
        "/v1/chunks" => ggml::chunks_handler(req).await,
        _ => error::invalid_endpoint(req.uri().path()),
    }
}

async fn handle_chat_embedding_request(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/v1/models" => ggml::models_handler().await,
        "/v1/chat/completions" => ggml::chat_completions_handler(req).await,
        "/v1/completions" => ggml::completions_handler(req).await,
        "/v1/embeddings" => ggml::embeddings_handler(req).await,
        "/v1/files" => ggml::files_handler(req).await,
        "/v1/chunks" => ggml::chunks_handler(req).await,
        _ => error::invalid_endpoint(req.uri().path()),
    }
}
