use serde::{Deserialize, Serialize};

// Document indexing request for JSON input
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentResult {
    pub filename: String,
    pub status: String,
    pub error: Option<String>,
}

// Index response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResponse {
    pub results: Vec<DocumentResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_url: Option<String>,
}

// Add these new structs for query handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    pub index: String,
}

fn default_top_k() -> usize {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub hits: Vec<SearchHit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub title: String,
    pub content: String,
    pub score: f32,
}
