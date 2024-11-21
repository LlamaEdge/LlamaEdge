//! Define types for the `rag` endpoint.

use serde::{Deserialize, Serialize};

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
