use serde::{Deserialize, Serialize};

// Document indexing request for JSON input
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexRequest {
    documents: Vec<DocumentInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInput {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
}

// Document processing result
#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentResult {
    filename: String,
    status: String,
    error: Option<String>,
}

// Index response
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexResponse {
    results: Vec<DocumentResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    download_url: Option<String>,
}
