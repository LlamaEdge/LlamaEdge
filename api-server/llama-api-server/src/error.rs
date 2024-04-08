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
        true => "500 Internal Server Error".to_string(),
        false => format!("500 Internal Server Error: {}", msg.as_ref()),
    };
    let mut response = Response::new(Body::from(err_msg));
    *response.status_mut() = hyper::StatusCode::INTERNAL_SERVER_ERROR;
    Ok(response)
}

pub(crate) fn bad_request(msg: impl AsRef<str>) -> Result<Response<Body>, hyper::Error> {
    let err_msg = match msg.as_ref().is_empty() {
        true => "400 Bad Request".to_string(),
        false => format!("400 Bad Request: {}", msg.as_ref()),
    };
    let mut response = Response::new(Body::from(err_msg));
    *response.status_mut() = hyper::StatusCode::BAD_REQUEST;
    Ok(response)
}

pub(crate) fn invalid_endpoint(msg: impl AsRef<str>) -> Result<Response<Body>, hyper::Error> {
    let err_msg = match msg.as_ref().is_empty() {
        true => "404 The requested service endpoint is not found".to_string(),
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
    /// Error returned while parsing socket address failed
    #[error("Failed to parse socket address: {0}")]
    SocketAddr(String),
    #[error("{0}")]
    Operation(String),
}
