//! Define types for the `rag` endpoint.

use crate::embeddings::EmbeddingRequest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagEmbeddingRequest {
    #[serde(rename = "embeddings")]
    pub embedding_request: EmbeddingRequest,
    #[serde(rename = "url")]
    pub qdrant_url: String,
    #[serde(rename = "collection_name")]
    pub qdrant_collection_name: String,
}
impl RagEmbeddingRequest {
    pub fn new(
        input: &[String],
        qdrant_url: impl AsRef<str>,
        qdrant_collection_name: impl AsRef<str>,
    ) -> Self {
        RagEmbeddingRequest {
            embedding_request: EmbeddingRequest {
                model: "dummy-embedding-model".to_string(),
                input: input.into(),
                encoding_format: None,
                user: None,
            },
            qdrant_url: qdrant_url.as_ref().to_string(),
            qdrant_collection_name: qdrant_collection_name.as_ref().to_string(),
        }
    }

    pub fn from_embedding_request(
        embedding_request: EmbeddingRequest,
        qdrant_url: impl AsRef<str>,
        qdrant_collection_name: impl AsRef<str>,
    ) -> Self {
        RagEmbeddingRequest {
            embedding_request,
            qdrant_url: qdrant_url.as_ref().to_string(),
            qdrant_collection_name: qdrant_collection_name.as_ref().to_string(),
        }
    }
}

#[test]
fn test_rag_serialize_embedding_request() {
    let embedding_request = EmbeddingRequest {
        model: "model".to_string(),
        input: "Hello, world!".into(),
        encoding_format: None,
        user: None,
    };
    let qdrant_url = "http://localhost:6333".to_string();
    let qdrant_collection_name = "qdrant_collection_name".to_string();
    let rag_embedding_request = RagEmbeddingRequest {
        embedding_request,
        qdrant_url,
        qdrant_collection_name,
    };
    let json = serde_json::to_string(&rag_embedding_request).unwrap();
    assert_eq!(
        json,
        r#"{"embeddings":{"model":"model","input":"Hello, world!"},"url":"http://localhost:6333","collection_name":"qdrant_collection_name"}"#
    );
}

#[test]
fn test_rag_deserialize_embedding_request() {
    let json = r#"{"embeddings":{"model":"model","input":["Hello, world!"]},"url":"http://localhost:6333","collection_name":"qdrant_collection_name"}"#;
    let rag_embedding_request: RagEmbeddingRequest = serde_json::from_str(json).unwrap();
    assert_eq!(rag_embedding_request.qdrant_url, "http://localhost:6333");
    assert_eq!(
        rag_embedding_request.qdrant_collection_name,
        "qdrant_collection_name"
    );
    assert_eq!(rag_embedding_request.embedding_request.model, "model");
    assert_eq!(
        rag_embedding_request.embedding_request.input,
        vec!["Hello, world!"].into()
    );
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunksRequest {
    pub id: String,
    pub filename: String,
    pub chunk_capacity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunksResponse {
    pub id: String,
    pub filename: String,
    pub chunks: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrieveObject {
    /// The retrieved sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub points: Option<Vec<RagScoredPoint>>,

    /// The number of similar points to retrieve
    pub limit: usize,

    /// The score threshold
    pub score_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagScoredPoint {
    /// Source of the context
    pub source: String,

    /// Points vector distance to the query vector
    pub score: f32,
}

#[test]
fn test_rag_serialize_retrieve_object() {
    {
        let ro = RetrieveObject {
            points: Some(vec![RagScoredPoint {
                source: "source".to_string(),
                score: 0.5,
            }]),
            limit: 1,
            score_threshold: 0.5,
        };
        let json = serde_json::to_string(&ro).unwrap();
        assert_eq!(
            json,
            r#"{"points":[{"source":"source","score":0.5}],"limit":1,"score_threshold":0.5}"#
        );
    }

    {
        let ro = RetrieveObject {
            points: None,
            limit: 1,
            score_threshold: 0.5,
        };
        let json = serde_json::to_string(&ro).unwrap();
        assert_eq!(json, r#"{"limit":1,"score_threshold":0.5}"#);
    }
}

#[test]
fn test_rag_deserialize_retrieve_object() {
    {
        let json =
            r#"{"points":[{"source":"source","score":0.5}],"limit":1,"score_threshold":0.5}"#;
        let ro: RetrieveObject = serde_json::from_str(json).unwrap();
        assert_eq!(ro.limit, 1);
        assert_eq!(ro.score_threshold, 0.5);
        assert!(ro.points.is_some());
        let points = ro.points.unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].source, "source");
        assert_eq!(points[0].score, 0.5);
    }

    {
        let json = r#"{"limit":1,"score_threshold":0.5}"#;
        let ro: RetrieveObject = serde_json::from_str(json).unwrap();
        assert_eq!(ro.limit, 1);
        assert_eq!(ro.score_threshold, 0.5);
        assert!(ro.points.is_none());
    }
}
