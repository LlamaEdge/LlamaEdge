use crate::utils::{LogLevel, NewLogRecord};
use hyper::{body::HttpBody, Body, Response};
use serde_json::json;
use thiserror::Error;

#[allow(dead_code)]
pub(crate) fn not_implemented() -> Response<Body> {
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

    Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .status(hyper::StatusCode::NOT_IMPLEMENTED)
        .body(Body::from("501 Not Implemented"))
        .unwrap()
}

pub(crate) fn internal_server_error(msg: impl AsRef<str>) -> Response<Body> {
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

    // log
    {
        let status_code = response.status();
        let response_version = format!("{:?}", response.version());
        let response_body_size: u64 = response.body().size_hint().lower();
        let response_status = status_code.as_u16();
        let response_is_informational = status_code.is_informational();
        let response_is_success = status_code.is_success();
        let response_is_redirection = status_code.is_redirection();
        let response_is_client_error = status_code.is_client_error();
        let response_is_server_error = status_code.is_server_error();
        let record = NewLogRecord::new(
            LogLevel::Error,
            None,
            json!({
                "version": response_version,
                "body_size": response_body_size,
                "status": response_status,
                "is_informational": response_is_informational,
                "is_success": response_is_success,
                "is_redirection": response_is_redirection,
                "is_client_error": response_is_client_error,
                "is_server_error": response_is_server_error,
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        error!(target: "response", "{}", message);
    }

    response
}

pub(crate) fn bad_request(msg: impl AsRef<str>) -> Response<Body> {
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

    Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .status(hyper::StatusCode::BAD_REQUEST)
        .body(Body::from(err_msg))
        .unwrap()
}

pub(crate) fn invalid_endpoint(msg: impl AsRef<str>) -> Response<Body> {
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

    Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .status(hyper::StatusCode::NOT_FOUND)
        .body(Body::from(err_msg))
        .unwrap()
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
