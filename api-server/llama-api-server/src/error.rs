use hyper::{Body, Response};
use thiserror::Error;

pub(crate) fn not_implemented() -> Result<Response<Body>, hyper::Error> {
    let mut response = Response::new(Body::from("501 Not Implemented"));
    *response.status_mut() = hyper::StatusCode::NOT_IMPLEMENTED;
    Ok(response)
}

pub(crate) fn internal_server_error(msg: impl AsRef<str>) -> Result<Response<Body>, hyper::Error> {
    let err_msg = match msg.as_ref().is_empty() {
        true => format!("500 Internal Server Error"),
        false => format!("500 Internal Server Error: {}", msg.as_ref()),
    };
    let mut response = Response::new(Body::from(err_msg));
    *response.status_mut() = hyper::StatusCode::INTERNAL_SERVER_ERROR;
    Ok(response)
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ServerError {
    #[error("Failed to parse socket address: {0}")]
    SocketAddr(String),
    #[error("Internal server error: {0}")]
    InternalServerError(String),
    #[error("Invalid prompt template type: {0}")]
    InvalidPromptTemplateType(String),
    #[error("Failed to set prompt context size. The `CTX_SIZE` is already set.")]
    PromptContextSize,
}
