use crate::{error::LlamaCoreError, ARCHIVES_DIR};
use base64::{engine::general_purpose, Engine as _};
use endpoints::files::{DeleteFileStatus, FileObject, ListFilesResponse};
use hyper::{body::to_bytes, Body, Method, Request};
use multipart::server::{Multipart, ReadEntry, ReadEntryResult};
use multipart_2021 as multipart;
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{Cursor, Read, Write},
    path::Path,
    time::SystemTime,
};
use walkdir::{DirEntry, WalkDir};

/// Upload a file and return the file object.
///
/// # Arguments
///
/// * `req`: The incoming POST request.
///
/// # Returns
///
/// A `FileObject` instance.
///
/// # Usage
///
/// This API should be used in a API server, which is responsible for
/// receiving the request and then forward the request to this API.
pub async fn upload_file(req: Request<Body>) -> Result<FileObject, LlamaCoreError> {
    if req.method() == Method::POST {
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
                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                return Err(LlamaCoreError::Operation(err_msg));
            }
        };

        let cursor = Cursor::new(body_bytes.to_vec());

        let mut multipart = Multipart::with_body(cursor, boundary.unwrap());

        while let ReadEntryResult::Entry(mut field) = multipart.read_entry_mut() {
            if &*field.headers.name == "file" {
                let filename = match field.headers.filename {
                    Some(filename) => filename,
                    None => {
                        let err_msg =
                            "Failed to upload the target file. The filename is not provided.";

                        // log
                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        return Err(LlamaCoreError::Operation(err_msg.to_string()));
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
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg));
                }

                let mut buffer = Vec::new();
                let size_in_bytes = match field.data.read_to_end(&mut buffer) {
                    Ok(size_in_bytes) => size_in_bytes,
                    Err(e) => {
                        let err_msg = format!("Failed to read the target file. {}", e);

                        // log
                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        return Err(LlamaCoreError::Operation(err_msg));
                    }
                };

                // create a unique file id
                let id = format!("file_{}", uuid::Uuid::new_v4());

                // save the file
                let path = Path::new(ARCHIVES_DIR);
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
                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        return Err(LlamaCoreError::Operation(err_msg));
                    }
                };
                file.write_all(&buffer[..]).unwrap();

                // log
                #[cfg(feature = "logging")]
                info!(target: "stdout", "file_id: {}, file_name: {}", &id, &filename);

                let created_at = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                    Ok(n) => n.as_secs(),
                    Err(_) => {
                        let err_msg = "Failed to get the current time.";

                        // log
                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        return Err(LlamaCoreError::Operation(err_msg.to_string()));
                    }
                };

                // create a file object
                return Ok(FileObject {
                    id,
                    bytes: size_in_bytes as u64,
                    created_at,
                    filename,
                    object: "file".to_string(),
                    purpose: "assistants".to_string(),
                });
            }
        }

        Err(LlamaCoreError::Operation(
            "Failed to upload the target file. No file is provided.".to_string(),
        ))
    } else {
        let err_msg = "Invalid HTTP Method.";

        // log
        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        Err(LlamaCoreError::Operation(err_msg.to_string()))
    }
}

/// Remove the target file by id.
///
/// # Arguments
///
/// * `id`: The id of the target file.
///
/// # Returns
///
/// A `DeleteFileStatus` instance.
pub fn remove_file(id: impl AsRef<str>) -> Result<DeleteFileStatus, LlamaCoreError> {
    let root = format!("{}/{}", ARCHIVES_DIR, id.as_ref());
    let status = match fs::remove_dir_all(root) {
        Ok(_) => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Successfully deleted the target file with id {}.", id.as_ref());

            DeleteFileStatus {
                id: id.as_ref().into(),
                object: "file".to_string(),
                deleted: true,
            }
        }
        Err(e) => {
            let err_msg = format!(
                "Failed to delete the target file with id {}. {}",
                id.as_ref(),
                e
            );

            // log
            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            DeleteFileStatus {
                id: id.as_ref().into(),
                object: "file".to_string(),
                deleted: false,
            }
        }
    };

    Ok(status)
}

/// List all files in the archives directory.
///
/// # Returns
///
/// A `ListFilesResponse` instance.
pub fn list_files() -> Result<ListFilesResponse, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Listing all archive files");

    let mut file_objects: Vec<FileObject> = Vec::new();
    for entry in WalkDir::new(ARCHIVES_DIR)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !is_hidden(&entry) && entry.path().is_file() {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "archive file: {}", entry.path().display());

            let id = entry
                .path()
                .parent()
                .and_then(|p| p.file_name())
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();

            let filename = entry
                .path()
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap()
                .to_string();

            let metadata = entry.path().metadata().unwrap();

            let created_at = metadata
                .created()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let bytes = metadata.len();

            let fo = FileObject {
                id,
                bytes,
                created_at,
                filename,
                object: "file".to_string(),
                purpose: "assistants".to_string(),
            };

            file_objects.push(fo);
        }
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Found {} archive files", file_objects.len());

    let file_objects = ListFilesResponse {
        object: "list".to_string(),
        data: file_objects,
    };

    Ok(file_objects)
}

/// Retrieve information about a specific file by id.
///
/// # Arguments
///
/// * `id`: The id of the target file.
///
/// # Returns
///
/// A `FileObject` instance.
pub fn retrieve_file(id: impl AsRef<str>) -> Result<FileObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Retrieving the target file with id {}", id.as_ref());

    let root = format!("{}/{}", ARCHIVES_DIR, id.as_ref());
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if !is_hidden(&entry) && entry.path().is_file() {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "archive file: {}", entry.path().display());

            let filename = entry
                .path()
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap()
                .to_string();

            let metadata = entry.path().metadata().unwrap();

            let created_at = metadata
                .created()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let bytes = metadata.len();

            return Ok(FileObject {
                id: id.as_ref().into(),
                bytes,
                created_at,
                filename,
                object: "file".to_string(),
                purpose: "assistants".to_string(),
            });
        }
    }

    Err(LlamaCoreError::FileNotFound)
}

/// Retrieve the content of a specific file by id.
///
/// # Arguments
///
/// * `id`: The id of the target file.
///
/// # Returns
///
/// A `Value` instance.
pub fn retrieve_file_content(id: impl AsRef<str>) -> Result<Value, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Retrieving the content of the target file with id {}", id.as_ref());

    let file_object = retrieve_file(id)?;
    let file_path = Path::new(ARCHIVES_DIR)
        .join(&file_object.id)
        .join(&file_object.filename);

    let base64_content = file_to_base64(&file_path)?;

    Ok(json!({
        "id": file_object.id,
        "bytes": file_object.bytes,
        "created_at": file_object.created_at,
        "filename": file_object.filename,
        "content": base64_content,
    }))
}

/// Download a specific file by id.
///
/// # Arguments
///
/// * `id`: The id of the target file.
///
/// # Returns
///
/// A tuple of `(String, Vec<u8>)`. The first element is the filename, and the second element is the file content.
pub fn download_file(id: impl AsRef<str>) -> Result<(String, Vec<u8>), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Downloading the target file with id {}", id.as_ref());

    let file_object = retrieve_file(id)?;
    let file_path = Path::new(ARCHIVES_DIR)
        .join(&file_object.id)
        .join(&file_object.filename);

    if !file_path.exists() {
        return Err(LlamaCoreError::FileNotFound);
    }

    // Open the file
    let mut file = match File::open(file_path) {
        Ok(file) => file,
        Err(e) => {
            let err_msg = format!("Failed to open the target file. {}", e);
            return Err(LlamaCoreError::Operation(err_msg));
        }
    };

    // read the file content as bytes
    let mut buffer = Vec::new();
    match file.read_to_end(&mut buffer) {
        Ok(_) => Ok((file_object.filename.clone(), buffer)),
        Err(e) => {
            let err_msg = format!("Failed to read the content of the target file. {}", e);

            // log
            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            Err(LlamaCoreError::Operation(err_msg))
        }
    }
}

fn is_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| s.starts_with("."))
        .unwrap_or(false)
}

fn file_to_base64(file_path: impl AsRef<Path>) -> Result<String, LlamaCoreError> {
    if !file_path.as_ref().exists() {
        return Err(LlamaCoreError::FileNotFound);
    }

    // Open the file
    let mut file = match File::open(file_path) {
        Ok(file) => file,
        Err(e) => {
            let err_msg = format!("Failed to open the target file. {}", e);
            return Err(LlamaCoreError::Operation(err_msg));
        }
    };

    // read the file content as bytes
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    Ok(general_purpose::STANDARD.encode(&buffer))
}
