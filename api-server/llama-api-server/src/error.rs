use hyper::{Body, Response};
use thiserror::Error;

#[allow(dead_code)]
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

pub(crate) fn invalid_endpoint(msg: impl AsRef<str>) -> Result<Response<Body>, hyper::Error> {
    let err_msg = match msg.as_ref().is_empty() {
        true => format!("404 The requested service endpoint is not found"),
        false => format!(
            "404 The requested service endpoint is not found: {}",
            msg.as_ref()
        ),
    };
    let mut response = Response::new(Body::from(err_msg));
    *response.status_mut() = hyper::StatusCode::NOT_FOUND;
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
    #[error("Failed to set `MAX_BUFFER_SIZE`. The `MAX_BUFFER_SIZE` is already set.")]
    MaxBufferSize,
    #[error("Failed to set `CTX_SIZE`. The `CTX_SIZE` is already set.")]
    ContextSize,
    #[error("Failed to set `METADATA`. The `METADATA` is already set.")]
    Metadata,
}
