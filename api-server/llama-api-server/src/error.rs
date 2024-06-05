use crate::utils::{LogLevel, NewLogRecord};
use hyper::{Body, Response};
use serde_json::json;
use thiserror::Error;

#[allow(dead_code)]
pub(crate) fn not_implemented() -> Result<Response<Body>, hyper::Error> {
    // log error
    {
        let record = NewLogRecord::new(
            LogLevel::Error,
            None,
            json!({
                "message": "501 Not Implemented",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        error!("{}", &message);
    }

    let response = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .status(hyper::StatusCode::NOT_IMPLEMENTED)
        .body(Body::from("501 Not Implemented"))
        .unwrap();

    Ok(response)
}

pub(crate) fn internal_server_error(msg: impl AsRef<str>) -> Result<Response<Body>, hyper::Error> {
    let err_msg = match msg.as_ref().is_empty() {
        true => "500 Internal Server Error".to_string(),
        false => format!("500 Internal Server Error: {}", msg.as_ref()),
    };

    // log error
    {
        let record = NewLogRecord::new(
            LogLevel::Error,
            None,
            json!({
                "message": format!("{msg}", msg = err_msg),
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        error!("{}", &message);
    }

    let response = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
        .body(Body::from(err_msg))
        .unwrap();

    Ok(response)
}

pub(crate) fn internal_server_error_new(msg: impl AsRef<str>) -> Response<Body> {
    let err_msg = match msg.as_ref().is_empty() {
        true => "500 Internal Server Error".to_string(),
        false => format!("500 Internal Server Error: {}", msg.as_ref()),
    };

    // log error
    {
        let record = NewLogRecord::new(
            LogLevel::Error,
            None,
            json!({
                "message": format!("{msg}", msg = err_msg),
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        error!("{}", &message);
    }

    Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
        .body(Body::from(err_msg))
        .unwrap()
}

pub(crate) fn bad_request(msg: impl AsRef<str>) -> Result<Response<Body>, hyper::Error> {
    let err_msg = match msg.as_ref().is_empty() {
        true => "400 Bad Request".to_string(),
        false => format!("400 Bad Request: {}", msg.as_ref()),
    };

    // log error
    {
        let record = NewLogRecord::new(
            LogLevel::Error,
            None,
            json!({
                "message": format!("{msg}", msg = err_msg),
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        error!("{}", &message);
    }

    let response = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .status(hyper::StatusCode::BAD_REQUEST)
        .body(Body::from(err_msg))
        .unwrap();

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

    // log error
    {
        let record = NewLogRecord::new(
            LogLevel::Error,
            None,
            json!({
                "message": format!("{msg}", msg = err_msg),
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        error!("{}", &message);
    }

    let response = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .status(hyper::StatusCode::NOT_FOUND)
        .body(Body::from(err_msg))
        .unwrap();

    Ok(response)
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ServerError {
    /// Error returned while parsing socket address failed
    #[error("Failed to parse socket address: {0}")]
    SocketAddr(String),
    /// Error returned while parsing CLI options failed
    #[error("{0}")]
    ArgumentError(String),
    /// Generic error returned while performing an operation
    #[error("{0}")]
    Operation(String),
}
