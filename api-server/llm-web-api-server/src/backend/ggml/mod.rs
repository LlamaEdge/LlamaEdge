pub(crate) mod llama;

use super::ServiceConfig;
use hyper::{Body, Request, Response};

use crate::error;

pub(crate) async fn handle_llama_request(
    req: Request<Body>,
    service_config: &ServiceConfig,
    model_name: impl AsRef<str>,
) -> Result<Response<Body>, hyper::Error> {
    println!("[BACKEND] service path: {}", service_config.path.as_str());

    match service_config.path.as_str() {
        "/v1/chat/completions" => {
            llama::llama_chat_completions_handler(req, model_name.as_ref()).await
        }
        "/v1/completions" => llama::llama_completions_handler().await,
        "/v1/embeddings" => llama::llama_embeddings_handler().await,
        "/v1/models" => llama::llama_models_handler().await,
        _ => error::not_implemented(),
    }
}
