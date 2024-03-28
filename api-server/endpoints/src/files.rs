//! Files are used to upload documents that can be used with features

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct FilesRequest {
    /// The File object (not file name) to be uploaded.
    file: FileObject,
    /// The intended purpose of the uploaded file.
    /// Use "fine-tune" for Fine-tuning and "assistants" for `Assistants` and `Messages`.
    purpose: String,
}

/// The File object represents a document that has been uploaded to the server.
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct FileObject {
    /// The file identifier, which can be referenced in the API endpoints.
    pub id: String,
    /// The size of the file, in bytes.
    pub bytes: u64,
    /// The Unix timestamp (in seconds) for when the file was created.
    pub created_at: u64,
    /// The name of the file.
    pub filename: String,
    /// The object type, which is always `file`.
    pub object: String,
    /// The intended purpose of the file. Supported values are `fine-tune`, `fine-tune-results`, `assistants`, and `assistants_output`.
    pub purpose: String,
}
