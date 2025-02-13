use crate::{error, utils::gen_chat_id, SERVER_INFO};
use endpoints::{
    audio::speech::SpeechRequest,
    chat::ChatCompletionRequest,
    completions::CompletionRequest,
    embeddings::{ChunksRequest, ChunksResponse, EmbeddingRequest},
    files::{DeleteFileStatus, FileObject},
};
use futures_util::TryStreamExt;
use hyper::{body::to_bytes, Body, Method, Request, Response};
use multipart::server::{Multipart, ReadEntry, ReadEntryResult};
use multipart_2021 as multipart;
use std::{
    fs::{self, File},
    io::{Cursor, Read, Write},
    path::Path,
    time::SystemTime,
};

/// List all models available.
pub(crate) async fn models_handler() -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming model list request.");

    let list_models_response = match llama_core::models::models().await {
        Ok(list_models_response) => list_models_response,
        Err(e) => {
            let err_msg = format!("Failed to get model list. Reason: {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // serialize response
    let s = match serde_json::to_string(&list_models_response) {
        Ok(s) => s,
        Err(e) => {
            let err_msg = format!("Failed to serialize the model list result. Reason: {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

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
    let res = match result {
        Ok(response) => response,
        Err(e) => {
            let err_msg = format!("Failed to get model list. Reason: {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    // log
    info!(target: "stdout", "Send the model list response.");

    res
}

/// Compute embeddings for the input text and return the embeddings object.
pub(crate) async fn embeddings_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming embeddings request");

    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body(Body::empty());

        match result {
            Ok(response) => return response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "embeddings_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };
    let mut embedding_request: EmbeddingRequest = match serde_json::from_slice(&body_bytes) {
        Ok(embedding_request) => embedding_request,
        Err(e) => {
            let mut err_msg = format!("Fail to deserialize embedding request: {}.", e);

            if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                err_msg = format!("{}\njson_value: {}", err_msg, json_value);
            }

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    if embedding_request.user.is_none() {
        embedding_request.user = Some(gen_chat_id())
    };
    let id = embedding_request.user.clone().unwrap();

    // log user id
    info!(target: "stdout", "user: {}", &id);

    let res = match llama_core::embeddings::embeddings(&embedding_request).await {
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
                        Ok(response) => response,
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            error!(target: "stdout", "{}", &err_msg);

                            error::internal_server_error(err_msg)
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Fail to serialize embedding object. {}", e);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = e.to_string();

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "stdout", "Send the embeddings response");

    res
}

/// Process a completion request and returns a completion response with the answer from the model.
pub(crate) async fn completions_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming completions request.");

    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body(Body::empty());

        match result {
            Ok(response) => return response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "completions_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };
    let mut completion_request: CompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(completion_request) => completion_request,
        Err(e) => {
            let mut err_msg = format!("Fail to deserialize completions request: {}.", e);

            if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                err_msg = format!("{}\njson_value: {}", err_msg, json_value);
            }

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    if completion_request.user.is_none() {
        completion_request.user = Some(gen_chat_id())
    };
    let id = completion_request.user.clone().unwrap();

    // log user id
    info!(target: "stdout", "user: {}", &id);

    let res = match llama_core::completions::completions(&completion_request).await {
        Ok(completion_object) => {
            // serialize completion object
            let s = match serde_json::to_string(&completion_object) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Fail to serialize completion object. {}", e);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

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
                Ok(response) => response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = e.to_string();

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "stdout", "Send the completions response.");

    res
}

/// Process a chat-completion request and returns a chat-completion response with the answer from the model.
pub(crate) async fn chat_completions_handler(mut req: Request<Body>) -> Response<Body> {
    info!(target: "stdout", "Handling the coming chat completion request");

    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body(Body::empty());

        match result {
            Ok(response) => return response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    info!(target: "stdout", "Prepare the chat completion request");

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };
    let mut chat_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chat_request) => chat_request,
        Err(e) => {
            let mut err_msg = format!("Fail to deserialize chat completion request: {}.", e);

            if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                err_msg = format!("{}\njson_value: {}", err_msg, json_value);
            }

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    // check if the user id is provided
    if chat_request.user.is_none() {
        chat_request.user = Some(gen_chat_id())
    };
    let id = chat_request.user.clone().unwrap();

    // log user id
    info!(target: "stdout", "user: {}", chat_request.user.clone().unwrap());

    debug!(target: "stdout", "request: {}", serde_json::to_string(&chat_request).unwrap());

    let res = match llama_core::chat::chat(&mut chat_request).await {
        Ok(result) => match result {
            either::Left(stream) => {
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
                        info!(target: "stdout", "finish chat completions in stream mode");

                        response
                    }
                    Err(e) => {
                        let err_msg =
                            format!("Failed chat completions in stream mode. Reason: {}", e);

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        error::internal_server_error(err_msg)
                    }
                }
            }
            either::Right(chat_completion_object) => {
                // serialize chat completion object
                let s = match serde_json::to_string(&chat_completion_object) {
                    Ok(s) => s,
                    Err(e) => {
                        let err_msg = format!("Failed to serialize chat completion object. {}", e);

                        // log
                        error!(target: "stdout", "{}", &err_msg);

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
                        info!(target: "stdout", "Finish chat completions in non-stream mode");

                        response
                    }
                    Err(e) => {
                        let err_msg =
                            format!("Failed chat completions in non-stream mode. Reason: {}", e);

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        error::internal_server_error(err_msg)
                    }
                }
            }
        },
        Err(e) => {
            let err_msg = format!("Failed to get chat completions. Reason: {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    // log
    info!(target: "stdout", "Send the chat completion response");

    res
}

/// Upload, download, retrieve and delete a file, or list all files.
///
/// - `POST /v1/files`: Upload a file.
/// - `GET /v1/files`: List all files.
/// - `GET /v1/files/{file_id}`: Retrieve a file by id.
/// - `GET /v1/files/{file_id}/content`: Retrieve the content of a file by id.
/// - `GET /v1/files/download/{file_id}`: Download a file by id.
/// - `DELETE /v1/files/{file_id}`: Delete a file by id.
///
pub(crate) async fn files_handler(req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming files request");

    let res = if req.method() == Method::POST {
        let boundary = "boundary=";

        let boundary = req.headers().get("content-type").and_then(|ct| {
            let ct = ct.to_str().ok()?;
            let idx = ct.find(boundary)?;
            Some(ct[idx + boundary.len()..].to_string())
        });

        let req_body = req.into_body();
        let body_bytes = match to_bytes(req_body).await {
            Ok(body_bytes) => body_bytes,
            Err(e) => {
                let err_msg = format!("Fail to read buffer from request body. {}", e);

                // log
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

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
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                if !((filename).to_lowercase().ends_with(".txt")
                    || (filename).to_lowercase().ends_with(".md")
                    || (filename).to_lowercase().ends_with(".png")
                    || (filename).to_lowercase().ends_with(".wav"))
                {
                    let err_msg = format!(
                        "Failed to upload the target file. Only files with 'txt', 'md', 'png', 'wav' extensions are supported. The file to be uploaded is {}.",
                        &filename
                    );

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }

                let mut buffer = Vec::new();
                let size_in_bytes = match field.data.read_to_end(&mut buffer) {
                    Ok(size_in_bytes) => size_in_bytes,
                    Err(e) => {
                        let err_msg = format!("Failed to read the target file. {}", e);

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                // create a unique file id
                let id = format!("file_{}", uuid::Uuid::new_v4());

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
                        let err_msg =
                            format!("Failed to create archive document {}. {}", &filename, e);

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };
                file.write_all(&buffer[..]).unwrap();

                // log
                info!(target: "stdout", "file_id: {}, file_name: {}", &id, &filename);

                let created_at = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                    Ok(n) => n.as_secs(),
                    Err(_) => {
                        let err_msg = "Failed to get the current time.";

                        // log
                        error!(target: "stdout", "{}", &err_msg);

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
                        error!(target: "stdout", "{}", &err_msg);

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
                    Ok(response) => response,
                    Err(e) => {
                        let err_msg = e.to_string();

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        error::internal_server_error(err_msg)
                    }
                }
            }
            None => {
                let err_msg = "Failed to upload the target file. Not found the target file.";

                // log
                error!(target: "stdout", "{}", &err_msg);

                error::internal_server_error(err_msg)
            }
        }
    } else if req.method() == Method::GET {
        let uri_path = req.uri().path().trim_end_matches('/').to_lowercase();

        // Split the path into segments
        let segments: Vec<&str> = uri_path.split('/').collect();

        match segments.as_slice() {
            ["", "v1", "files"] => list_files(),
            ["", "v1", "files", file_id, "content"] => {
                if !file_id.starts_with("file_") {
                    let err_msg = format!("unsupported uri path: {}", uri_path);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }

                retrieve_file_content(file_id)
            }
            ["", "v1", "files", file_id] => {
                if !file_id.starts_with("file_") {
                    let err_msg = format!("unsupported uri path: {}", uri_path);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }

                retrieve_file(file_id)
            }
            ["", "v1", "files", "download", file_id] => download_file(file_id),
            _ => {
                let err_msg = format!("unsupported uri path: {}", uri_path);

                // log
                error!(target: "stdout", "{}", &err_msg);

                error::internal_server_error(err_msg)
            }
        }
    } else if req.method() == Method::DELETE {
        let id = req.uri().path().trim_start_matches("/v1/files/");
        let status = match llama_core::files::remove_file(id) {
            Ok(status) => status,
            Err(e) => {
                let err_msg = format!("Failed to delete the target file with id {}. {}", id, e);

                // log
                error!(target: "stdout", "{}", &err_msg);

                DeleteFileStatus {
                    id: id.into(),
                    object: "file".to_string(),
                    deleted: false,
                }
            }
        };

        // serialize status
        let s = match serde_json::to_string(&status) {
            Ok(s) => s,
            Err(e) => {
                let err_msg = format!(
                    "Failed to serialize the status of the file deletion operation. {}",
                    e
                );

                // log
                error!(target: "stdout", "{}", &err_msg);

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
            Ok(response) => response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "stdout", "{}", &err_msg);

                error::internal_server_error(err_msg)
            }
        }
    } else if req.method() == Method::OPTIONS {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body(Body::empty());

        match result {
            Ok(response) => return response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "files_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    } else {
        let err_msg = "Invalid HTTP Method.";

        // log
        error!(target: "stdout", "{}", &err_msg);

        error::internal_server_error(err_msg)
    };

    info!(target: "stdout", "Send the files response");

    res
}

fn list_files() -> Response<Body> {
    match llama_core::files::list_files() {
        Ok(file_objects) => {
            // serialize chat completion object
            let s = match serde_json::to_string(&file_objects) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Failed to serialize file list. {}", e);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

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
                Ok(response) => response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("Failed to list all files. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    }
}

fn retrieve_file(id: impl AsRef<str>) -> Response<Body> {
    match llama_core::files::retrieve_file(id) {
        Ok(fo) => {
            // serialize chat completion object
            let s = match serde_json::to_string(&fo) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Failed to serialize file object. {}", e);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

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
                Ok(response) => response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("{}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    }
}

fn retrieve_file_content(id: impl AsRef<str>) -> Response<Body> {
    match llama_core::files::retrieve_file_content(id) {
        Ok(content) => {
            // serialize chat completion object
            let s = match serde_json::to_string(&content) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Failed to serialize file content. {}", e);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

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
                Ok(response) => response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("{}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    }
}

fn download_file(id: impl AsRef<str>) -> Response<Body> {
    info!(target: "stdout", "download file: {}", id.as_ref());

    match llama_core::files::download_file(id) {
        Ok((filename, buffer)) => {
            // get the extension of the file
            let extension = filename.split('.').next_back().unwrap_or("unknown");
            let content_type = match extension {
                "txt" => "text/plain",
                "json" => "application/json",
                "png" => "image/png",
                "jpg" => "image/jpeg",
                "jpeg" => "image/jpeg",
                "wav" => "audio/wav",
                "mp3" => "audio/mpeg",
                "mp4" => "video/mp4",
                "md" => "text/markdown",
                _ => {
                    let err_msg = format!("Unsupported file extension: {}", extension);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }
            };
            let content_disposition = format!("attachment; filename={}", filename);

            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .header("Content-Type", content_type)
                .header("Content-Disposition", content_disposition)
                .body(Body::from(buffer));

            match result {
                Ok(response) => response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("{}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    }
}

/// Segment the text into chunks and return the chunks response.
pub(crate) async fn chunks_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming chunks request");

    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body(Body::empty());

        match result {
            Ok(response) => return response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "chunks_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    let chunks_request: ChunksRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chunks_request) => chunks_request,
        Err(e) => {
            let mut err_msg = format!("Fail to deserialize chunks request: {}.", e);

            if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                err_msg = format!("{}\njson_value: {}", err_msg, json_value);
            }

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    // check if the archives directory exists
    let path = Path::new("archives");
    if !path.exists() {
        let err_msg = "The `archives` directory does not exist.";

        // log
        error!(target: "stdout", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    }

    // check if the archive id exists
    let archive_path = path.join(&chunks_request.id);
    if !archive_path.exists() {
        let err_msg = format!("Not found archive id: {}", &chunks_request.id);

        // log
        error!(target: "stdout", "{}", &err_msg);

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
        error!(target: "stdout", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    }

    // log
    info!(target: "stdout", "file_id: {}, file_name: {}", &chunks_request.id, &chunks_request.filename);

    // get the extension of the archived file
    let extension = match file_path.extension().and_then(std::ffi::OsStr::to_str) {
        Some(extension) => extension,
        None => {
            let err_msg = format!(
                "Failed to get the extension of the archived `{}`.",
                &chunks_request.filename
            );

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // open the file
    let mut file = match File::open(&file_path) {
        Ok(file) => file,
        Err(e) => {
            let err_msg = format!("Failed to open `{}`. {}", &chunks_request.filename, e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // read the file
    let mut contents = String::new();
    if let Err(e) = file.read_to_string(&mut contents) {
        let err_msg = format!("Failed to read `{}`. {}", &chunks_request.filename, e);

        // log
        error!(target: "stdout", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    }

    let res = match llama_core::embeddings::chunk_text(
        &contents,
        extension,
        chunks_request.chunk_capacity,
    ) {
        Ok(chunks) => {
            let chunks_response = ChunksResponse {
                id: chunks_request.id,
                filename: chunks_request.filename,
                chunks,
            };

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
                        Ok(response) => response,
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            error!(target: "stdout", "{}", &err_msg);

                            error::internal_server_error(err_msg)
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Fail to serialize chunks response. {}", e);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = e.to_string();

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "stdout", "Send the chunks response.");

    res
}

/// Return the server info.
pub(crate) async fn server_info_handler() -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming server info request.");

    // get the server info
    let server_info = match SERVER_INFO.get() {
        Some(server_info) => server_info,
        None => {
            let err_msg = "The server info is not set.";

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error("The server info is not set.");
        }
    };

    // serialize server info
    let s = match serde_json::to_string(&server_info) {
        Ok(s) => s,
        Err(e) => {
            let err_msg = format!("Fail to serialize server info. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

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
    let res = match result {
        Ok(response) => response,
        Err(e) => {
            let err_msg = e.to_string();

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "stdout", "Send the server info response.");

    res
}

/// Create the audio speech.
pub(crate) async fn audio_speech_handler(req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming audio speech request.");

    if req.method().eq(&Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body(Body::empty());

        match result {
            Ok(response) => return response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    info!(target: "stdout", "Prepare the chat completion request.");

    // parse request
    let body_bytes = match to_bytes(req.into_body()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };
    let speech_request: SpeechRequest = match serde_json::from_slice(&body_bytes) {
        Ok(speech_request) => speech_request,
        Err(e) => {
            let err_msg = format!("Fail to deserialize speech request: {msg}", msg = e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    let audio_buffer = match llama_core::tts::create_speech(speech_request).await {
        Ok(buffer) => buffer,
        Err(e) => {
            let err_msg = format!("Failed to transcribe the audio. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .header("Content-Type", "audio/wav")
        .header("Content-Disposition", "attachment; filename=audio.wav")
        .body(Body::from(audio_buffer));

    let res = match result {
        Ok(response) => response,
        Err(e) => {
            let err_msg = e.to_string();

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "stdout", "Send the audio speech response");

    res
}
