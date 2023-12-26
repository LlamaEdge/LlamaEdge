pub(crate) mod ggml;

use crate::{error, ModelInfo};
use chat_prompts::PromptTemplateType;
use hyper::{Body, Request, Response};

pub(crate) async fn handle_llama_request(
    req: Request<Body>,
    model_info: ModelInfo,
    template_ty: PromptTemplateType,
    created: u64,
    log_prompts: bool,
    stream: bool,
) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/v1/chat/completions" => {
            ggml::chat_completions_handler(req, template_ty, log_prompts, stream).await
        }
        "/v1/completions" => ggml::completions_handler(req).await,
        // "/v1/embeddings" => ggml::_embeddings_handler().await,
        "/v1/models" => ggml::models_handler(model_info, template_ty, created).await,
        _ => error::invalid_endpoint(req.uri().path()),
    }
}
