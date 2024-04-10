pub(crate) mod ggml;

use crate::error;
use hyper::{Body, Request, Response};

pub(crate) async fn handle_llama_request(
    req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/v1/chat/completions" => ggml::chat_completions_handler(req).await,
        "/v1/completions" => ggml::completions_handler(req).await,
        "/v1/models" => ggml::models_handler().await,
        "/v1/embeddings" => ggml::embeddings_handler(req).await,
        "/v1/files" => ggml::files_handler(req).await,
        "/v1/chunks" => ggml::chunks_handler(req).await,
        _ => error::invalid_endpoint(req.uri().path()),
    }
}
