use crate::{
    error,
    utils::{gen_chat_id, LogLevel, NewLogRecord},
    SERVER_INFO,
};
use endpoints::{
    chat::ChatCompletionRequest,
    completions::CompletionRequest,
    embeddings::EmbeddingRequest,
    files::FileObject,
    rag::{ChunksRequest, ChunksResponse},
};
use futures_util::TryStreamExt;
use hyper::{
    body::{to_bytes, HttpBody},
    Body, Method, Request, Response,
};
use multipart::server::{Multipart, ReadEntry, ReadEntryResult};
use multipart_2021 as multipart;
use serde_json::json;
use std::{
    fs::{self, File},
    io::{Cursor, Read, Write},
    path::Path,
    time::SystemTime,
};

/// List all models available.
pub(crate) async fn models_handler() -> Result<Response<Body>, hyper::Error> {
    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "Handle model list request",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "models_handler", "{}", message);
    }

    let list_models_response = match llama_core::models::models().await {
        Ok(list_models_response) => list_models_response,
        Err(e) => {
            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": format!("Failed to get model list. Reason: {}", e.to_string()),
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "models_handler", "{}", message);
            }
            return error::internal_server_error(e.to_string());
        }
    };

    // serialize response
    let s = match serde_json::to_string(&list_models_response) {
        Ok(s) => s,
        Err(e) => {
            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": format!("Failed to serialize the model list result. Reason: {}", e.to_string()),
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "models_handler", "{}", message);
            }
            return error::internal_server_error(e.to_string());
        }
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .header("Content-Type", "application/json")
        .body(Body::from(s));
    match result {
        Ok(response) => {
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
                    LogLevel::Info,
                    None,
                    json!({
                        "response_version": response_version,
                        "response_body_size": response_body_size,
                        "response_status": response_status,
                        "response_is_informational": response_is_informational,
                        "response_is_success": response_is_success,
                        "response_is_redirection": response_is_redirection,
                        "response_is_client_error": response_is_client_error,
                        "response_is_server_error": response_is_server_error,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                info!(target: "models_handler", "{}", message);
            }
            Ok(response)
        }
        Err(e) => {
            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": format!("Failed to get model list. Reason: {}", e.to_string()),
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "models_handler", "{}", message);
            }
            error::internal_server_error(e.to_string())
        }
    }
}

/// Compute embeddings for the input text and return the embeddings object.
pub(crate) async fn embeddings_handler(
    mut req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "Handle embeddings request",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "embeddings_handler", "{}", message);
    }

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut embedding_request: EmbeddingRequest = match serde_json::from_slice(&body_bytes) {
        Ok(embedding_request) => embedding_request,
        Err(e) => {
            let err_msg = format!("Fail to deserialize embedding request: {msg}", msg = e);

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "embeddings_handler", "{}", message);
            }

            return error::bad_request(err_msg);
        }
    };

    if embedding_request.user.is_none() {
        embedding_request.user = Some(gen_chat_id())
    };
    let id = embedding_request.user.clone().unwrap();

    // log user id
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "user": &id,
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "embedding_request", "{}", message);
    }

    println!("\n[+] Running embeddings handler ...");
    match llama_core::embeddings::embeddings(&embedding_request).await {
        Ok(embedding_response) => {
            // serialize embedding object
            match serde_json::to_string(&embedding_response) {
                Ok(s) => {
                    // return response
                    let result = Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .header("Content-Type", "application/json")
                        .header("user", id)
                        .body(Body::from(s));
                    match result {
                        Ok(response) => {
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
                                    LogLevel::Info,
                                    None,
                                    json!({
                                        "response_version": response_version,
                                        "response_body_size": response_body_size,
                                        "response_status": response_status,
                                        "response_is_informational": response_is_informational,
                                        "response_is_success": response_is_success,
                                        "response_is_redirection": response_is_redirection,
                                        "response_is_client_error": response_is_client_error,
                                        "response_is_server_error": response_is_server_error,
                                    }),
                                );
                                let message = serde_json::to_string(&record).unwrap();
                                info!(target: "embeddings_handler", "{}", message);
                            }

                            Ok(response)
                        }
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            {
                                let record = NewLogRecord::new(
                                    LogLevel::Error,
                                    None,
                                    json!({
                                        "message": &err_msg,
                                    }),
                                );
                                let message = serde_json::to_string(&record).unwrap();
                                error!(target: "embeddings_handler", "{}", message);
                            }

                            error::internal_server_error(err_msg)
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Fail to serialize embedding object. {}", e);

                    // log
                    {
                        let record = NewLogRecord::new(
                            LogLevel::Error,
                            None,
                            json!({
                                "message": &err_msg,
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        error!(target: "embeddings_handler", "{}", message);
                    }

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = e.to_string();

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "embeddings_handler", "{}", message);
            }

            error::internal_server_error(err_msg)
        }
    }
}

pub(crate) async fn completions_handler(
    mut req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "Handle completions request",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "completions_handler", "{}", message);
    }

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut completion_request: CompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(completion_request) => completion_request,
        Err(e) => {
            let err_msg = format!("Fail to deserialize completions request: {msg}", msg = e);

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "completions_handler", "{}", message);
            }

            return error::bad_request(err_msg);
        }
    };

    if completion_request.user.is_none() {
        completion_request.user = Some(gen_chat_id())
    };
    let id = completion_request.user.clone().unwrap();

    // log user id
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "user": &id,
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "completions_handler", "{}", message);
    }

    println!("\n[+] Running completions handler ...");
    match llama_core::completions::completions(&completion_request).await {
        Ok(completion_object) => {
            // serialize completion object
            let s = match serde_json::to_string(&completion_object) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Fail to serialize completion object. {}", e);

                    // log
                    {
                        let record = NewLogRecord::new(
                            LogLevel::Error,
                            None,
                            json!({
                                "message": &err_msg,
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        error!(target: "completions_handler", "{}", message);
                    }

                    return error::internal_server_error(err_msg);
                }
            };

            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .header("Content-Type", "application/json")
                .header("user", id)
                .body(Body::from(s));
            match result {
                Ok(response) => {
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
                            LogLevel::Info,
                            None,
                            json!({
                                "response_version": response_version,
                                "response_body_size": response_body_size,
                                "response_status": response_status,
                                "response_is_informational": response_is_informational,
                                "response_is_success": response_is_success,
                                "response_is_redirection": response_is_redirection,
                                "response_is_client_error": response_is_client_error,
                                "response_is_server_error": response_is_server_error,
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        info!(target: "completions_handler", "{}", message);
                    }

                    Ok(response)
                }
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    {
                        let record = NewLogRecord::new(
                            LogLevel::Error,
                            None,
                            json!({
                                "message": &err_msg,
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        error!(target: "completions_handler", "{}", message);
                    }

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = e.to_string();

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "completions_handler", "{}", message);
            }

            error::internal_server_error(err_msg)
        }
    }
}

/// Process a chat-completion request and returns a chat-completion response with the answer from the model.
pub(crate) async fn chat_completions_handler(
    mut req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "Handle chat completion request",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "chat_completions_handler", "{}", message);
    }

    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body(Body::empty());

        match result {
            Ok(response) => return Ok(response),
            Err(e) => {
                return error::internal_server_error(e.to_string());
            }
        }
    }

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut chat_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chat_request) => chat_request,
        Err(e) => {
            let err_msg = format!(
                "Fail to deserialize chat completion request: {msg}",
                msg = e
            );

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "chat_completions_handler", "{}", message);
            }

            return error::bad_request(err_msg);
        }
    };

    // check if the user id is provided
    if chat_request.user.is_none() {
        chat_request.user = Some(gen_chat_id())
    };

    // log user id
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "user": chat_request.user.clone().unwrap(),
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "chat_completions_handler", "{}", message);
    }

    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "Handle chat completion request",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "chat_completions_handler", "{}", message);
    }

    // handle chat request
    let response = match chat_request.stream {
        Some(true) => chat_completions_stream(chat_request).await,
        Some(false) | None => chat_completions(chat_request).await,
    };

    // log response
    let status_code = response.status();
    if status_code.as_u16() < 400 {
        // log response
        let response_version = format!("{:?}", response.version());
        let response_body_size: u64 = response.body().size_hint().lower();
        let response_status = status_code.as_u16();
        let response_is_informational = status_code.is_informational();
        let response_is_success = status_code.is_success();
        let response_is_redirection = status_code.is_redirection();
        let response_is_client_error = status_code.is_client_error();
        let response_is_server_error = status_code.is_server_error();
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "response_version": response_version,
                "response_body_size": response_body_size,
                "response_status": response_status,
                "response_is_informational": response_is_informational,
                "response_is_success": response_is_success,
                "response_is_redirection": response_is_redirection,
                "response_is_client_error": response_is_client_error,
                "response_is_server_error": response_is_server_error,
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "chat_completions_handler", "{}", message);
    } else {
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
                "response_version": response_version,
                "response_body_size": response_body_size,
                "response_status": response_status,
                "response_is_informational": response_is_informational,
                "response_is_success": response_is_success,
                "response_is_redirection": response_is_redirection,
                "response_is_client_error": response_is_client_error,
                "response_is_server_error": response_is_server_error,
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "chat_completions_handler", "{}", message);
    }

    Ok(response)
}

/// Process a chat-completion request in stream mode and returns a chat-completion response with the answer from the model.
async fn chat_completions_stream(mut chat_request: ChatCompletionRequest) -> Response<Body> {
    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "start chat completions in stream mode",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "chat_completions_stream", "{}", message);
    }

    let id = chat_request.user.clone().unwrap();
    match llama_core::chat::chat_completions_stream(&mut chat_request).await {
        Ok(stream) => {
            let stream = stream.map_err(|e| e.to_string());

            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .header("Content-Type", "text/event-stream")
                .header("Cache-Control", "no-cache")
                .header("Connection", "keep-alive")
                .header("user", id)
                .body(Body::wrap_stream(stream));

            match result {
                Ok(response) => {
                    // log
                    {
                        let record = NewLogRecord::new(
                            LogLevel::Info,
                            None,
                            json!({
                                "message": "finish chat completions in stream mode",
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        info!(target: "chat_completions_stream", "{}", message);
                    }
                    response
                }
                Err(e) => {
                    // log
                    {
                        let record = NewLogRecord::new(
                            LogLevel::Error,
                            None,
                            json!({
                                "message": format!("Failed chat completions in stream mode. Reason: {}", e.to_string()),
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        error!(target: "chat_completions_stream", "{}", message);
                    }
                    error::internal_server_error_new(e.to_string())
                }
            }
        }
        Err(e) => {
            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": format!("Failed chat completions in stream mode. Reason: {}", e.to_string()),
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "chat_completions_stream", "{}", message);
            }
            error::internal_server_error_new(e.to_string())
        }
    }
}

/// Process a chat-completion request and returns a chat-completion response with the answer from the model.
async fn chat_completions(mut chat_request: ChatCompletionRequest) -> Response<Body> {
    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "start chat completions in non-stream mode",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "chat_completions", "{}", message);
    }

    let id = chat_request.user.clone().unwrap();
    match llama_core::chat::chat_completions(&mut chat_request).await {
        Ok(chat_completion_object) => {
            // serialize chat completion object
            let s = match serde_json::to_string(&chat_completion_object) {
                Ok(s) => s,
                Err(e) => {
                    // log
                    {
                        let record = NewLogRecord::new(
                            LogLevel::Error,
                            None,
                            json!({
                                "message": format!(
                                    "Fail to serialize chat completion object. {}",
                                    e
                                ),
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        error!(target: "chat_completions", "{}", message);
                    }
                    return error::internal_server_error_new(format!(
                        "Fail to serialize chat completion object. {}",
                        e
                    ));
                }
            };

            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .header("Content-Type", "application/json")
                .header("user", id)
                .body(Body::from(s));

            match result {
                Ok(response) => {
                    // log
                    {
                        let record = NewLogRecord::new(
                            LogLevel::Info,
                            None,
                            json!({
                                "message": "finish chat completions in non-stream mode",
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        info!(target: "chat_completions", "{}", message);
                    }
                    response
                }
                Err(e) => {
                    // log
                    {
                        let record = NewLogRecord::new(
                            LogLevel::Error,
                            None,
                            json!({
                                "message": format!("Failed chat completions in non-stream mode. Reason: {}", e.to_string()),
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        error!(target: "chat_completions", "{}", message);
                    }
                    error::internal_server_error_new(e.to_string())
                }
            }
        }
        Err(e) => {
            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": format!("Failed chat completions in non-stream mode. Reason: {}", e.to_string()),
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "chat_completions", "{}", message);
            }
            error::internal_server_error_new(e.to_string())
        }
    }
}

pub(crate) async fn files_handler(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "Handle files request",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "files_handler", "{}", message);
    }

    if req.method() == Method::POST {
        let boundary = "boundary=";

        let boundary = req.headers().get("content-type").and_then(|ct| {
            let ct = ct.to_str().ok()?;
            let idx = ct.find(boundary)?;
            Some(ct[idx + boundary.len()..].to_string())
        });

        let req_body = req.into_body();
        let body_bytes = to_bytes(req_body).await?;
        let cursor = Cursor::new(body_bytes.to_vec());

        let mut multipart = Multipart::with_body(cursor, boundary.unwrap());

        let mut file_object: Option<FileObject> = None;
        while let ReadEntryResult::Entry(mut field) = multipart.read_entry_mut() {
            if &*field.headers.name == "file" {
                let filename = match field.headers.filename {
                    Some(filename) => filename,
                    None => {
                        let err_msg =
                            "Failed to upload the target file. The filename is not provided.";

                        // log
                        {
                            let record = NewLogRecord::new(
                                LogLevel::Error,
                                None,
                                json!({
                                    "message": &err_msg,
                                }),
                            );
                            let message = serde_json::to_string(&record).unwrap();
                            error!(target: "files_handler", "{}", message);
                        }

                        return error::internal_server_error(err_msg);
                    }
                };

                if !((filename).to_lowercase().ends_with(".txt")
                    || (filename).to_lowercase().ends_with(".md"))
                {
                    let err_msg = format!(
                        "Failed to upload the target file. Only files with 'txt' and 'md' extensions are supported. The file extension is {}.",
                        &filename
                    );

                    // log
                    {
                        let record = NewLogRecord::new(
                            LogLevel::Error,
                            None,
                            json!({
                                "message": &err_msg,
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        error!(target: "files_handler", "{}", message);
                    }

                    return error::internal_server_error(err_msg);
                }

                let mut buffer = Vec::new();
                let size_in_bytes = match field.data.read_to_end(&mut buffer) {
                    Ok(size_in_bytes) => size_in_bytes,
                    Err(e) => {
                        let err_msg = format!("Failed to read the target file. {}", e);

                        // log
                        {
                            let record = NewLogRecord::new(
                                LogLevel::Error,
                                None,
                                json!({
                                    "message": &err_msg,
                                }),
                            );
                            let message = serde_json::to_string(&record).unwrap();
                            error!(target: "files_handler", "{}", message);
                        }

                        return error::internal_server_error(err_msg);
                    }
                };

                // create a unique file id
                let id = format!("file_{}", uuid::Uuid::new_v4());

                // log
                {
                    let record = NewLogRecord::new(
                        LogLevel::Info,
                        None,
                        json!({
                            "file_id": &id,
                            "file_name": &filename,
                        }),
                    );
                    let message = serde_json::to_string(&record).unwrap();
                    info!(target: "files_handler", "{}", message);
                }

                // save the file
                let path = Path::new("archives");
                if !path.exists() {
                    fs::create_dir(path).unwrap();
                }
                let file_path = path.join(&id);
                if !file_path.exists() {
                    fs::create_dir(&file_path).unwrap();
                }
                let mut file = match File::create(file_path.join(&filename)) {
                    Ok(file) => file,
                    Err(e) => {
                        return error::internal_server_error(format!(
                            "Failed to create archive document {}. {}",
                            &filename, e
                        ));
                    }
                };
                file.write_all(&buffer[..]).unwrap();

                let created_at = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                    Ok(n) => n.as_secs(),
                    Err(_) => {
                        let err_msg = "Failed to get the current time.";

                        // log
                        {
                            let record = NewLogRecord::new(
                                LogLevel::Error,
                                None,
                                json!({
                                    "message": &err_msg,
                                }),
                            );
                            let message = serde_json::to_string(&record).unwrap();
                            error!(target: "files_handler", "{}", message);
                        }

                        return error::internal_server_error(err_msg);
                    }
                };

                // create a file object
                file_object = Some(FileObject {
                    id,
                    bytes: size_in_bytes as u64,
                    created_at,
                    filename,
                    object: "file".to_string(),
                    purpose: "assistants".to_string(),
                });

                break;
            }
        }

        match file_object {
            Some(fo) => {
                // serialize chat completion object
                let s = match serde_json::to_string(&fo) {
                    Ok(s) => s,
                    Err(e) => {
                        let err_msg = format!("Failed to serialize file object. {}", e);

                        // log
                        {
                            let record = NewLogRecord::new(
                                LogLevel::Error,
                                None,
                                json!({
                                    "message": &err_msg,
                                }),
                            );
                            let message = serde_json::to_string(&record).unwrap();
                            error!(target: "files_handler", "{}", message);
                        }

                        return error::internal_server_error(err_msg);
                    }
                };

                // return response
                let result = Response::builder()
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "*")
                    .header("Access-Control-Allow-Headers", "*")
                    .header("Content-Type", "application/json")
                    .body(Body::from(s));

                match result {
                    Ok(response) => {
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
                                LogLevel::Info,
                                None,
                                json!({
                                    "response_version": response_version,
                                    "response_body_size": response_body_size,
                                    "response_status": response_status,
                                    "response_is_informational": response_is_informational,
                                    "response_is_success": response_is_success,
                                    "response_is_redirection": response_is_redirection,
                                    "response_is_client_error": response_is_client_error,
                                    "response_is_server_error": response_is_server_error,
                                }),
                            );
                            let message = serde_json::to_string(&record).unwrap();
                            info!(target: "files_handler", "{}", message);
                        }

                        Ok(response)
                    }
                    Err(e) => {
                        let err_msg = e.to_string();

                        // log
                        {
                            let record = NewLogRecord::new(
                                LogLevel::Error,
                                None,
                                json!({
                                    "message": &err_msg,
                                }),
                            );
                            let message = serde_json::to_string(&record).unwrap();
                            error!(target: "files_handler", "{}", message);
                        }

                        error::internal_server_error(err_msg)
                    }
                }
            }
            None => {
                let err_msg = "Failed to upload the target file. Not found the target file.";

                // log
                {
                    let record = NewLogRecord::new(
                        LogLevel::Error,
                        None,
                        json!({
                            "message": &err_msg,
                        }),
                    );
                    let message = serde_json::to_string(&record).unwrap();
                    error!(target: "files_handler", "{}", message);
                }

                error::internal_server_error(err_msg)
            }
        }
    } else if req.method() == Method::GET {
        let err_msg = "Not implemented for listing files.";

        // log
        {
            let record = NewLogRecord::new(
                LogLevel::Error,
                None,
                json!({
                    "message": &err_msg,
                }),
            );
            let message = serde_json::to_string(&record).unwrap();
            error!(target: "files_handler", "{}", message);
        }

        error::internal_server_error(err_msg)
    } else {
        let err_msg = "Invalid HTTP Method.";

        // log
        {
            let record = NewLogRecord::new(
                LogLevel::Error,
                None,
                json!({
                    "message": &err_msg,
                }),
            );
            let message = serde_json::to_string(&record).unwrap();
            error!(target: "files_handler", "{}", message);
        }

        error::internal_server_error(err_msg)
    }
}

pub(crate) async fn chunks_handler(mut req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "Handle chunks request",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "chunks_handler", "{}", message);
    }

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let chunks_request: ChunksRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chunks_request) => chunks_request,
        Err(e) => {
            let err_msg = format!("Fail to deserialize chunks request: {msg}", msg = e);

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "chunks_handler", "{}", message);
            }

            return error::bad_request(err_msg);
        }
    };

    // check if the archives directory exists
    let path = Path::new("archives");
    if !path.exists() {
        let err_msg = "The `archives` directory does not exist.";

        // log
        {
            let record = NewLogRecord::new(
                LogLevel::Error,
                None,
                json!({
                    "message": &err_msg,
                }),
            );
            let message = serde_json::to_string(&record).unwrap();
            error!(target: "chunks_handler", "{}", message);
        }

        return error::internal_server_error(err_msg);
    }

    // check if the archive id exists
    let archive_path = path.join(&chunks_request.id);
    if !archive_path.exists() {
        let err_msg = format!("Not found archive id: {}", &chunks_request.id);

        // log
        {
            let record = NewLogRecord::new(
                LogLevel::Error,
                None,
                json!({
                    "message": &err_msg,
                }),
            );
            let message = serde_json::to_string(&record).unwrap();
            error!(target: "chunks_handler", "{}", message);
        }

        return error::internal_server_error(err_msg);
    }

    // check if the file exists
    let file_path = archive_path.join(&chunks_request.filename);
    if !file_path.exists() {
        let err_msg = format!(
            "Not found file: {} in archive id: {}",
            &chunks_request.filename, &chunks_request.id
        );

        // log
        {
            let record = NewLogRecord::new(
                LogLevel::Error,
                None,
                json!({
                    "message": &err_msg,
                }),
            );
            let message = serde_json::to_string(&record).unwrap();
            error!(target: "chunks_handler", "{}", message);
        }

        return error::internal_server_error(err_msg);
    }

    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "file_id": &chunks_request.id,
                "file_name": &chunks_request.filename,
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "chunks_handler", "{}", message);
    }

    // get the extension of the archived file
    let extension = match file_path.extension().and_then(std::ffi::OsStr::to_str) {
        Some(extension) => extension,
        None => {
            let err_msg = format!(
                "Failed to get the extension of the archived `{}`.",
                &chunks_request.filename
            );

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "chunks_handler", "{}", message);
            }

            return error::internal_server_error(err_msg);
        }
    };

    // open the file
    let mut file = match File::open(&file_path) {
        Ok(file) => file,
        Err(e) => {
            let err_msg = format!("Failed to open `{}`. {}", &chunks_request.filename, e);

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "chunks_handler", "{}", message);
            }

            return error::internal_server_error(err_msg);
        }
    };

    // read the file
    let mut contents = String::new();
    if let Err(e) = file.read_to_string(&mut contents) {
        let err_msg = format!("Failed to read `{}`. {}", &chunks_request.filename, e);

        // log
        {
            let record = NewLogRecord::new(
                LogLevel::Error,
                None,
                json!({
                    "message": &err_msg,
                }),
            );
            let message = serde_json::to_string(&record).unwrap();
            error!(target: "chunks_handler", "{}", message);
        }

        return error::internal_server_error(err_msg);
    }

    match llama_core::rag::chunk_text(&contents, extension, chunks_request.chunk_capacity) {
        Ok(chunks) => {
            let chunks_response = ChunksResponse {
                id: chunks_request.id,
                filename: chunks_request.filename,
                chunks,
            };

            println!("[+] File chunked successfully.\n");

            // serialize embedding object
            match serde_json::to_string(&chunks_response) {
                Ok(s) => {
                    // return response
                    let result = Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .header("Content-Type", "application/json")
                        .body(Body::from(s));
                    match result {
                        Ok(response) => {
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
                                    LogLevel::Info,
                                    None,
                                    json!({
                                        "response_version": response_version,
                                        "response_body_size": response_body_size,
                                        "response_status": response_status,
                                        "response_is_informational": response_is_informational,
                                        "response_is_success": response_is_success,
                                        "response_is_redirection": response_is_redirection,
                                        "response_is_client_error": response_is_client_error,
                                        "response_is_server_error": response_is_server_error,
                                    }),
                                );
                                let message = serde_json::to_string(&record).unwrap();
                                info!(target: "chunks_handler", "{}", message);
                            }

                            Ok(response)
                        }
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            {
                                let record = NewLogRecord::new(
                                    LogLevel::Error,
                                    None,
                                    json!({
                                        "message": &err_msg,
                                    }),
                                );
                                let message = serde_json::to_string(&record).unwrap();
                                error!(target: "chunks_handler", "{}", message);
                            }

                            error::internal_server_error(err_msg)
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Fail to serialize chunks response. {}", e);

                    // log
                    {
                        let record = NewLogRecord::new(
                            LogLevel::Error,
                            None,
                            json!({
                                "message": &err_msg,
                            }),
                        );
                        let message = serde_json::to_string(&record).unwrap();
                        error!(target: "chunks_handler", "{}", message);
                    }

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = e.to_string();

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "chunks_handler", "{}", message);
            }

            error::internal_server_error(err_msg)
        }
    }
}

pub(crate) async fn server_info() -> Result<Response<Body>, hyper::Error> {
    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "Handle server inforequest",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!(target: "server_info", "{}", message);
    }

    // get the server info
    let server_info = match SERVER_INFO.get() {
        Some(server_info) => server_info,
        None => {
            let err_msg = "The server info is not set.";

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "chunks_handler", "{}", message);
            }

            return error::internal_server_error("The server info is not set.");
        }
    };

    // serialize server info
    let s = match serde_json::to_string(&server_info) {
        Ok(s) => s,
        Err(e) => {
            let err_msg = format!("Fail to serialize server info. {}", e);

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "chunks_handler", "{}", message);
            }

            return error::internal_server_error(err_msg);
        }
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .header("Content-Type", "application/json")
        .body(Body::from(s));
    match result {
        Ok(response) => {
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
                    LogLevel::Info,
                    None,
                    json!({
                        "response_version": response_version,
                        "response_body_size": response_body_size,
                        "response_status": response_status,
                        "response_is_informational": response_is_informational,
                        "response_is_success": response_is_success,
                        "response_is_redirection": response_is_redirection,
                        "response_is_client_error": response_is_client_error,
                        "response_is_server_error": response_is_server_error,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                info!(target: "server_info", "{}", message);
            }

            Ok(response)
        }
        Err(e) => {
            let err_msg = e.to_string();

            // log
            {
                let record = NewLogRecord::new(
                    LogLevel::Error,
                    None,
                    json!({
                        "message": &err_msg,
                    }),
                );
                let message = serde_json::to_string(&record).unwrap();
                error!(target: "chunks_handler", "{}", message);
            }

            error::internal_server_error(err_msg)
        }
    }
}
