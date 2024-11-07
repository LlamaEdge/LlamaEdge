use crate::error::LlamaCoreError;
use endpoints::files::{DeleteFileStatus, FileObject, ListFilesResponse};
use std::fs;
use walkdir::{DirEntry, WalkDir};

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
    let root = format!("archives/{}", id.as_ref());
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
    let root = "archives";
    let mut file_objects: Vec<FileObject> = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
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
    let root = format!("archives/{}", id.as_ref());
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

fn is_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| s.starts_with("."))
        .unwrap_or(false)
}
