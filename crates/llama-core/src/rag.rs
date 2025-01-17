//! Define APIs for RAG operations.

use crate::{embeddings::embeddings, error::LlamaCoreError, running_mode, RunningMode};
use endpoints::{
    embeddings::{EmbeddingObject, EmbeddingRequest, EmbeddingsResponse, InputText},
    rag::{RagScoredPoint, RetrieveObject},
};
use qdrant::*;
use serde_json::Value;
use std::collections::HashSet;

/// Convert document chunks to embeddings.
///
/// # Arguments
///
/// * `embedding_request` - A reference to an `EmbeddingRequest` object.
///
/// # Returns
///
/// Name of the Qdrant collection if successful.
pub async fn rag_doc_chunks_to_embeddings(
    embedding_request: &EmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Convert document chunks to embeddings.");

    let running_mode = running_mode()?;
    if running_mode != RunningMode::Rag {
        let err_msg = format!(
            "Creating knowledge base is not supported in the {} mode.",
            running_mode
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    let qdrant_url = match embedding_request.vdb_server_url.as_deref() {
        Some(url) => url.to_string(),
        None => {
            let err_msg = "The VectorDB server URL is not provided.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };
    let qdrant_collection_name = match embedding_request.vdb_collection_name.as_deref() {
        Some(name) => name.to_string(),
        None => {
            let err_msg = "The VectorDB collection name is not provided.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute embeddings for document chunks.");

    #[cfg(feature = "logging")]
    if let Ok(request_str) = serde_json::to_string(&embedding_request) {
        debug!(target: "stdout", "Embedding request: {}", request_str);
    }

    // compute embeddings for the document
    let response = embeddings(embedding_request).await?;
    let embeddings = response.data.as_slice();
    let dim = embeddings[0].embedding.len();

    // create a Qdrant client
    let mut qdrant_client = qdrant::Qdrant::new_with_url(qdrant_url);

    // set the API key if provided
    if let Some(key) = embedding_request.vdb_api_key.as_deref() {
        if !key.is_empty() {
            #[cfg(feature = "logging")]
            debug!(target: "stdout", "Set the API key for the VectorDB server.");

            qdrant_client.set_api_key(key);
        }
    }

    // create a collection
    qdrant_create_collection(&qdrant_client, &qdrant_collection_name, dim).await?;

    let chunks = match &embedding_request.input {
        InputText::String(text) => vec![text.clone()],
        InputText::ArrayOfStrings(texts) => texts.clone(),
        InputText::ArrayOfTokens(tokens) => tokens.iter().map(|t| t.to_string()).collect(),
        InputText::ArrayOfTokenArrays(token_arrays) => token_arrays
            .iter()
            .map(|tokens| tokens.iter().map(|t| t.to_string()).collect())
            .collect(),
    };

    // create and upsert points
    qdrant_persist_embeddings(
        &qdrant_client,
        &qdrant_collection_name,
        embeddings,
        chunks.as_slice(),
    )
    .await?;

    Ok(response)
}

/// Convert a query to embeddings.
///
/// # Arguments
///
/// * `embedding_request` - A reference to an `EmbeddingRequest` object.
pub async fn rag_query_to_embeddings(
    embedding_request: &EmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute embeddings for the user query.");

    let running_mode = running_mode()?;
    if running_mode != RunningMode::Rag {
        let err_msg = format!("The RAG query is not supported in the {running_mode} mode.",);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    embeddings(embedding_request).await
}

/// Retrieve similar points from the Qdrant server using the query embedding
///
/// # Arguments
///
/// * `query_embedding` - A reference to a query embedding.
///
/// * `qdrant_url` - URL of the Qdrant server.
///
/// * `qdrant_collection_name` - Name of the Qdrant collection to be created.
///
/// * `limit` - Number of retrieved results.
///
/// * `score_threshold` - The minimum score of the retrieved results.
pub async fn rag_retrieve_context(
    query_embedding: &[f32],
    vdb_server_url: impl AsRef<str>,
    vdb_collection_name: impl AsRef<str>,
    limit: usize,
    score_threshold: Option<f32>,
    vdb_api_key: Option<String>,
) -> Result<RetrieveObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    {
        info!(target: "stdout", "Retrieve context.");

        info!(target: "stdout", "qdrant_url: {}, qdrant_collection_name: {}, limit: {}, score_threshold: {}", vdb_server_url.as_ref(), vdb_collection_name.as_ref(), limit, score_threshold.unwrap_or_default());
    }

    let running_mode = running_mode()?;
    if running_mode != RunningMode::Rag {
        let err_msg = format!(
            "The context retrieval is not supported in the {} mode.",
            running_mode
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    // create a Qdrant client
    let mut qdrant_client = qdrant::Qdrant::new_with_url(vdb_server_url.as_ref().to_string());

    // set the API key if provided
    if let Some(key) = vdb_api_key.as_deref() {
        if !key.is_empty() {
            #[cfg(feature = "logging")]
            debug!(target: "stdout", "Set the API key for the VectorDB server.");

            qdrant_client.set_api_key(key);
        }
    }

    // search for similar points
    let scored_points = match qdrant_search_similar_points(
        &qdrant_client,
        vdb_collection_name.as_ref(),
        query_embedding,
        limit,
        score_threshold,
    )
    .await
    {
        Ok(points) => points,
        Err(e) => {
            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", e.to_string());

            return Err(e);
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "remove duplicates from {} scored points", scored_points.len());

    // remove duplicates, which have the same source
    let mut seen = HashSet::new();
    let unique_scored_points: Vec<ScoredPoint> = scored_points
        .into_iter()
        .filter(|point| {
            seen.insert(
                point
                    .payload
                    .as_ref()
                    .unwrap()
                    .get("source")
                    .unwrap()
                    .to_string(),
            )
        })
        .collect();

    #[cfg(feature = "logging")]
    info!(target: "stdout", "number of unique scored points: {}", unique_scored_points.len());

    let ro = match unique_scored_points.is_empty() {
        true => RetrieveObject {
            points: None,
            limit,
            score_threshold: score_threshold.unwrap_or(0.0),
        },
        false => {
            let mut points: Vec<RagScoredPoint> = vec![];
            for point in unique_scored_points.iter() {
                if let Some(payload) = &point.payload {
                    if let Some(source) = payload.get("source").and_then(Value::as_str) {
                        points.push(RagScoredPoint {
                            source: source.to_string(),
                            score: point.score,
                        })
                    }

                    // For debugging purpose, log the optional search field if it exists
                    #[cfg(feature = "logging")]
                    if let Some(search) = payload.get("search").and_then(Value::as_str) {
                        info!(target: "stdout", "search: {}", search);
                    }
                }
            }

            RetrieveObject {
                points: Some(points),
                limit,
                score_threshold: score_threshold.unwrap_or(0.0),
            }
        }
    };

    Ok(ro)
}

async fn qdrant_create_collection(
    qdrant_client: &qdrant::Qdrant,
    collection_name: impl AsRef<str>,
    dim: usize,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Create a Qdrant collection named {} of {} dimensions.", collection_name.as_ref(), dim);

    if let Err(e) = qdrant_client
        .create_collection(collection_name.as_ref(), dim as u32)
        .await
    {
        let err_msg = e.to_string();

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Qdrant(err_msg));
    }

    Ok(())
}

async fn qdrant_persist_embeddings(
    qdrant_client: &qdrant::Qdrant,
    collection_name: impl AsRef<str>,
    embeddings: &[EmbeddingObject],
    chunks: &[String],
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Persist embeddings to the Qdrant instance.");

    let mut points = Vec::<Point>::new();
    for embedding in embeddings {
        // convert the embedding to a vector
        let vector: Vec<_> = embedding.embedding.iter().map(|x| *x as f32).collect();

        // create a payload
        let payload = serde_json::json!({"source": chunks[embedding.index as usize]})
            .as_object()
            .map(|m| m.to_owned());

        // create a point
        let p = Point {
            id: PointId::Num(embedding.index),
            vector,
            payload,
        };

        points.push(p);
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Number of points to be upserted: {}", points.len());

    if let Err(e) = qdrant_client
        .upsert_points(collection_name.as_ref(), points)
        .await
    {
        let err_msg = format!("{}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Qdrant(err_msg));
    }

    Ok(())
}

async fn qdrant_search_similar_points(
    qdrant_client: &qdrant::Qdrant,
    collection_name: impl AsRef<str>,
    query_vector: &[f32],
    limit: usize,
    score_threshold: Option<f32>,
) -> Result<Vec<ScoredPoint>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Search similar points from the qdrant instance.");

    match qdrant_client
        .search_points(
            collection_name.as_ref(),
            query_vector.to_vec(),
            limit as u64,
            score_threshold,
        )
        .await
    {
        Ok(search_result) => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Number of similar points found: {}", search_result.len());

            Ok(search_result)
        }
        Err(e) => {
            let err_msg = e.to_string();

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            Err(LlamaCoreError::Qdrant(err_msg))
        }
    }
}
