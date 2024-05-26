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
    // log request
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "Handle model list request",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!("{}", &message);
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
                error!("{}", &message);
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
                error!("{}", &message);
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
                info!("{}", &message);
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
                error!("{}", &message);
            }
            error::internal_server_error(e.to_string())
        }
    }
}

/// Compute embeddings for the input text and return the embeddings object.
pub(crate) async fn embeddings_handler(
    mut req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut embedding_request: EmbeddingRequest = match serde_json::from_slice(&body_bytes) {
        Ok(embedding_request) => embedding_request,
        Err(e) => {
            return error::bad_request(format!("Fail to parse embedding request: {msg}", msg = e));
        }
    };

    if embedding_request.user.is_none() {
        embedding_request.user = Some(gen_chat_id())
    };
    let id = embedding_request.user.clone().unwrap();

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
                        Ok(response) => Ok(response),
                        Err(e) => error::internal_server_error(e.to_string()),
                    }
                }
                Err(e) => error::internal_server_error(format!(
                    "Fail to serialize embedding object. {}",
                    e
                )),
            }
        }
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

pub(crate) async fn completions_handler(
    mut req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut completion_request: CompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(completion_request) => completion_request,
        Err(e) => {
            return error::bad_request(format!(
                "Failed to deserialize completion request. {msg}",
                msg = e
            ));
        }
    };

    if completion_request.user.is_none() {
        completion_request.user = Some(gen_chat_id())
    };
    let id = completion_request.user.clone().unwrap();

    println!("\n[+] Running completions handler ...");
    match llama_core::completions::completions(&completion_request).await {
        Ok(completion_object) => {
            // serialize completion object
            let s = match serde_json::to_string(&completion_object) {
                Ok(s) => s,
                Err(e) => {
                    return error::internal_server_error(format!(
                        "Failed to serialize completion object. {msg}",
                        msg = e
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
                Ok(response) => Ok(response),
                Err(e) => {
                    println!("[*] Error: {}", e);
                    error::internal_server_error(e.to_string())
                }
            }
        }
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

/// Process a chat-completion request and returns a chat-completion response with the answer from the model.
pub(crate) async fn chat_completions_handler(
    mut req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
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

    // log request
    {
        let method = hyper::http::Method::as_str(req.method()).to_string();
        let path = req.uri().path().to_string();
        let version = format!("{:?}", req.version());
        let size: u64 = req
            .headers()
            .get("content-length")
            .unwrap()
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "request_method": method,
                "request_path": path,
                "request_http_version": version,
                "request_body_size": size,
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!("{}", &message);
    }

    // log
    {
        let record = NewLogRecord::new(
            LogLevel::Info,
            None,
            json!({
                "message": "Parse chat completion request body",
            }),
        );
        let message = serde_json::to_string(&record).unwrap();
        info!("{}", &message);
    }

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut chat_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chat_request) => chat_request,
        Err(e) => {
            return error::bad_request(format!(
                "Fail to parse chat completion request: {msg}",
                msg = e
            ));
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
        info!("{}", &message);
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
        info!("{}", &message);
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
        info!("{}", &message);
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
        error!("{}", &message);
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
        info!("{}", &message);
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
                        info!("{}", &message);
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
                        error!("{}", &message);
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
                error!("{}", &message);
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
        info!("{}", &message);
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
                        error!("{}", &message);
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
                        info!("{}", &message);
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
                        error!("{}", &message);
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
                error!("{}", &message);
            }
            error::internal_server_error_new(e.to_string())
        }
    }
}

pub(crate) async fn files_handler(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    if req.method() == Method::POST {
        println!("\n[+] Running files handler ...");

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
                        return error::internal_server_error(
                            "Failed to upload the target file. The filename is not provided.",
                        );
                    }
                };

                if !((filename).to_lowercase().ends_with(".txt")
                    || (filename).to_lowercase().ends_with(".md"))
                {
                    return error::internal_server_error(
                        "Failed to upload the target file. Only files with 'txt' and 'md' extensions are supported.",
                    );
                }

                let mut buffer = Vec::new();
                let size_in_bytes = match field.data.read_to_end(&mut buffer) {
                    Ok(size_in_bytes) => size_in_bytes,
                    Err(e) => {
                        return error::internal_server_error(format!(
                            "Failed to read the target file. {}",
                            e
                        ));
                    }
                };

                // create a unique file id
                let id = format!("file_{}", uuid::Uuid::new_v4());

                println!("    * Saving to {}/{}", &id, &filename);

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
                        return error::internal_server_error("Failed to get the current time.")
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

        println!("[+] File uploaded successfully.\n");
        match file_object {
            Some(fo) => {
                // serialize chat completion object
                let s = match serde_json::to_string(&fo) {
                    Ok(s) => s,
                    Err(e) => {
                        return error::internal_server_error(format!(
                            "Fail to serialize file object. {}",
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
                    .body(Body::from(s));

                match result {
                    Ok(response) => Ok(response),
                    Err(e) => error::internal_server_error(e.to_string()),
                }
            }
            None => error::internal_server_error(
                "Failed to upload the target file. Not found the target file.",
            ),
        }
    } else if req.method() == Method::GET {
        error::internal_server_error("Not implemented for listing files.")
    } else {
        error::internal_server_error("Invalid HTTP Method.")
    }
}

pub(crate) async fn chunks_handler(mut req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    println!("\n[+] Running chunks handler ...");

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let chunks_request: ChunksRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chunks_request) => chunks_request,
        Err(e) => {
            return error::bad_request(format!("Fail to parse chunks request: {msg}", msg = e));
        }
    };

    println!("[+] Detecting the target file ...");
    // check if the archives directory exists
    let path = Path::new("archives");
    if !path.exists() {
        return error::internal_server_error("The `archives` directory does not exist.");
    }

    // check if the archive id exists
    let archive_path = path.join(&chunks_request.id);
    if !archive_path.exists() {
        let message = format!("Not found archive id: {}", &chunks_request.id);
        return error::internal_server_error(message);
    }

    // check if the file exists
    let file_path = archive_path.join(&chunks_request.filename);
    if !file_path.exists() {
        let message = format!(
            "Not found file: {} in archive id: {}",
            &chunks_request.filename, &chunks_request.id
        );
        return error::internal_server_error(message);
    }
    println!(
        "    * Found {}/{}",
        &chunks_request.id, &chunks_request.filename
    );

    // get the extension of the archived file
    let extension = match file_path.extension().and_then(std::ffi::OsStr::to_str) {
        Some(extension) => extension,
        None => {
            return error::internal_server_error(format!(
                "Failed to get the extension of the archived `{}`.",
                &chunks_request.filename
            ));
        }
    };

    // open the file
    let mut file = match File::open(&file_path) {
        Ok(file) => file,
        Err(e) => {
            return error::internal_server_error(format!(
                "Failed to open `{}`. {}",
                &chunks_request.filename, e
            ));
        }
    };

    // read the file
    let mut contents = String::new();
    if let Err(e) = file.read_to_string(&mut contents) {
        return error::internal_server_error(format!(
            "Failed to read `{}`. {}",
            &chunks_request.filename, e
        ));
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
                        Ok(response) => Ok(response),
                        Err(e) => error::internal_server_error(e.to_string()),
                    }
                }
                Err(e) => error::internal_server_error(format!(
                    "Fail to serialize chunks response. {}",
                    e
                )),
            }
        }
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

pub(crate) async fn server_info() -> Result<Response<Body>, hyper::Error> {
    // get the server info
    let server_info = match SERVER_INFO.get() {
        Some(server_info) => server_info,
        None => {
            return error::internal_server_error("The server info is not set.");
        }
    };

    // serialize server info
    let s = match serde_json::to_string(&server_info) {
        Ok(s) => s,
        Err(e) => {
            return error::internal_server_error(format!("Fail to serialize server info. {}", e));
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
        Ok(response) => Ok(response),
        Err(e) => error::internal_server_error(e.to_string()),
    }
}
