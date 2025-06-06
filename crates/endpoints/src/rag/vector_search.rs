//! Define types for the `rag` endpoint.

use crate::embeddings::EmbeddingsResponse;
#[cfg(feature = "index")]
use crate::rag::keyword_search::IndexResponse;
use serde::{Deserialize, Serialize};

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
    pub score: f64,
    /// The source of the context
    pub from: DataFrom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataFrom {
    /// The context is from the vector database.
    #[serde(rename = "vector_search")]
    VectorSearch,
    /// The context is from the keyword search.
    #[serde(rename = "keyword_search")]
    KeywordSearch,
}

#[test]
fn test_rag_serialize_retrieve_object() {
    {
        let ro = RetrieveObject {
            points: Some(vec![RagScoredPoint {
                source: "source".to_string(),
                score: 0.5,
                from: DataFrom::VectorSearch,
            }]),
            limit: 1,
            score_threshold: 0.5,
        };
        let json = serde_json::to_string(&ro).unwrap();
        assert_eq!(
            json,
            r#"{"points":[{"source":"source","score":0.5,"from":"vector_search"}],"limit":1,"score_threshold":0.5}"#
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
        let json = r#"{"points":[{"source":"source","score":0.5,"from":"vector_search"}],"limit":1,"score_threshold":0.5}"#;
        let ro: RetrieveObject = serde_json::from_str(json).unwrap();
        assert_eq!(ro.limit, 1);
        assert_eq!(ro.score_threshold, 0.5);
        assert!(ro.points.is_some());
        let points = ro.points.unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].source, "source");
        assert_eq!(points[0].score, 0.5);
        assert_eq!(points[0].from, DataFrom::VectorSearch);
    }

    {
        let json = r#"{"limit":1,"score_threshold":0.5}"#;
        let ro: RetrieveObject = serde_json::from_str(json).unwrap();
        assert_eq!(ro.limit, 1);
        assert_eq!(ro.score_threshold, 0.5);
        assert!(ro.points.is_none());
    }
}

/// Defines the response of rag creation.
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRagResponse {
    #[cfg(feature = "index")]
    #[serde(rename = "index", skip_serializing_if = "Option::is_none")]
    pub index_response: Option<IndexResponse>,
    #[serde(rename = "embeddings")]
    pub embeddings_response: EmbeddingsResponse,
}
