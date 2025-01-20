use serde::{Deserialize, Serialize};

// Document indexing request for JSON input
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexRequest {
    pub documents: Vec<DocumentInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInput {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

// Document processing result
#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentResult {
    pub filename: String,
    pub status: String,
    pub error: Option<String>,
}

// Index response
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexResponse {
    pub results: Vec<DocumentResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_url: Option<String>,
}
