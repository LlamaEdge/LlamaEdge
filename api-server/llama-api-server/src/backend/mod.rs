pub(crate) mod ggml;

use crate::error;
use chat_prompts::PromptTemplateType;
use hyper::{Body, Request, Response};

pub(crate) async fn handle_llama_request(
    req: Request<Body>,
    model_name: impl AsRef<str>,
    template_ty: PromptTemplateType,
    created: u64,
    metadata: String,
) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/v1/chat/completions" => {
            ggml::chat_completions_handler(req, model_name.as_ref(), template_ty, metadata).await
        }
        "/v1/completions" => ggml::completions_handler(req, model_name.as_ref(), metadata).await,
        // "/v1/embeddings" => ggml::_embeddings_handler().await,
        "/v1/models" => ggml::models_handler(template_ty, created).await,
        _ => error::invalid_endpoint(req.uri().path()),
    }
}
