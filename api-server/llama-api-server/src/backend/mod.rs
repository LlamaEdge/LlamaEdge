pub(crate) mod ggml;

use crate::error;
use chat_prompts::PromptTemplateType;
use hyper::{Body, Request, Response};

pub(crate) async fn handle_llama_request(
    req: Request<Body>,
    template_ty: PromptTemplateType,
    log_prompts: bool,
) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/v1/chat/completions" => {
            ggml::chat_completions_handler(req, template_ty, log_prompts).await
        }
        "/v1/completions" => ggml::completions_handler(req).await,
        // "/v1/embeddings" => ggml::_embeddings_handler().await,
        "/v1/models" => ggml::models_handler().await,
        _ => error::invalid_endpoint(req.uri().path()),
    }
}
