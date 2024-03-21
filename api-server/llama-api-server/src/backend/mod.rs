pub(crate) mod ggml;

use crate::{error, QDRANT_CONFIG};
use chat_prompts::PromptTemplateType;
use hyper::{Body, Request, Response};

pub(crate) async fn handle_llama_request(
    req: Request<Body>,
    template_ty: PromptTemplateType,
    log_prompts: bool,
) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/v1/chat/completions" => match QDRANT_CONFIG.get() {
            Some(_) => ggml::rag_query_handler(req, template_ty, log_prompts).await,
            None => ggml::chat_completions_handler(req, template_ty, log_prompts).await,
        },
        "/v1/completions" => ggml::completions_handler(req).await,
        "/v1/models" => ggml::models_handler().await,
        "/v1/embeddings" => match QDRANT_CONFIG.get() {
            Some(_) => ggml::rag_doc_chunks_to_embeddings2_handler(req, log_prompts).await,
            None => ggml::embeddings_handler(req).await,
        },
        "/v1/files" => ggml::files_handler(req).await,
        "/v1/chunks" => ggml::chunks_handler(req).await,
        _ => error::invalid_endpoint(req.uri().path()),
    }
}
