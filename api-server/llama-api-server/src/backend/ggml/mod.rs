pub(crate) mod llama;

use crate::error;
use hyper::{Body, Request, Response};
use prompt::PromptTemplateType;

pub(crate) async fn handle_llama_request(
    req: Request<Body>,
    model_name: impl AsRef<str>,
    template_ty: PromptTemplateType,
    created: u64,
) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/v1/chat/completions" => {
            llama::llama_chat_completions_handler(req, model_name.as_ref(), template_ty).await
        }
        // "/v1/completions" => llama::llama_completions_handler().await,
        // "/v1/embeddings" => llama::llama_embeddings_handler().await,
        "/v1/models" => llama::llama_models_handler(created).await,
        _ => error::not_implemented(),
    }
}
