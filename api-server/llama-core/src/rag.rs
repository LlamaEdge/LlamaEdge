use crate::{embeddings::embeddings, error::LlamaCoreError};
use endpoints::{
    embeddings::{EmbeddingObject, EmbeddingsResponse},
    rag::RagEmbeddingRequest,
};
use qdrant::*;

/// Convert document chunks to embeddings.
///
/// # Arguments
///
/// * `embedding_request` - A reference to an `EmbeddingRequest` object.
///
/// * `qdrant_url` - URL of the Qdrant server.
///
/// * `qdrant_collection_name` - Name of the Qdrant collection to be created.
pub async fn rag_doc_chunks_to_embeddings(
    rag_embedding_request: &RagEmbeddingRequest,
) -> Result<(), LlamaCoreError> {
    let embedding_request = &rag_embedding_request.embedding_request;
    let qdrant_url = rag_embedding_request.qdrant_url.as_str();
    let qdrant_collection_name = rag_embedding_request.qdrant_collection_name.as_str();

    // compute embeddings for the document
    let response = embeddings(embedding_request).await?;

    let chunks = embedding_request.input.as_slice();
    let embeddings = response.data.as_slice();
    let dim = embeddings[0].embedding.len();

    // create a Qdrant client
    let qdrant_client = qdrant::Qdrant::new_with_url(qdrant_url.to_string());

    // create a collection
    qdrant_create_collection(&qdrant_client, qdrant_collection_name, dim).await?;

    // create and upsert points
    qdrant_persist_embeddings(&qdrant_client, qdrant_collection_name, embeddings, chunks).await?;

    Ok(())
}

/// Convert a query to embeddings.
///
/// # Arguments
///
/// * `embedding_request` - A reference to an `EmbeddingRequest` object.
pub async fn rag_query_to_embeddings(
    rag_embedding_request: &RagEmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    embeddings(&rag_embedding_request.embedding_request).await
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
/// * `limit` - Max number of retrieved result.
pub async fn rag_retrieve_context(
    query_embedding: &[f32],
    qdrant_url: impl AsRef<str>,
    qdrant_collection_name: impl AsRef<str>,
    limit: usize,
) -> Result<Vec<ScoredPoint>, LlamaCoreError> {
    // create a Qdrant client
    let qdrant_client = qdrant::Qdrant::new_with_url(qdrant_url.as_ref().to_string());

    // search for similar points
    let search_result = qdrant_search_similar_points(
        &qdrant_client,
        qdrant_collection_name.as_ref(),
        query_embedding,
        limit,
    )
    .await
    .map_err(|e| LlamaCoreError::Operation(e))?;

    Ok(search_result)
}

async fn qdrant_create_collection(
    qdrant_client: &qdrant::Qdrant,
    collection_name: impl AsRef<str>,
    dim: usize,
) -> Result<(), LlamaCoreError> {
    println!("[+] Creating a collection ...");
    // let collection_name = "my_test";
    println!("    * Collection name: {}", collection_name.as_ref());
    println!("    * Dimension: {}", dim);
    if let Err(err) = qdrant_client
        .create_collection(collection_name.as_ref(), dim as u32)
        .await
    {
        println!("Failed to create collection. {}", err.to_string());
        return Err(LlamaCoreError::Operation(err.to_string()));
    }

    Ok(())
}

async fn qdrant_persist_embeddings(
    qdrant_client: &qdrant::Qdrant,
    collection_name: impl AsRef<str>,
    embeddings: &[EmbeddingObject],
    chunks: &[String],
) -> Result<(), LlamaCoreError> {
    println!("[+] Creating points to save embeddings ...");
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

    // upsert points
    println!("[+] Upserting points ...");
    if let Err(err) = qdrant_client
        .upsert_points(collection_name.as_ref(), points)
        .await
    {
        println!("Failed to upsert points. {}", err.to_string());
        return Err(LlamaCoreError::Operation(err.to_string()));
    }

    Ok(())
}

async fn qdrant_search_similar_points(
    qdrant_client: &qdrant::Qdrant,
    collection_name: impl AsRef<str>,
    query_vector: &[f32],
    limit: usize,
) -> Result<Vec<ScoredPoint>, String> {
    println!("[+] Searching for similar points ...");
    let search_result = qdrant_client
        .search_points(
            collection_name.as_ref(),
            query_vector.to_vec(),
            limit as u64,
        )
        .await;

    Ok(search_result)
}

/// Type alias for `qdrant::ScoredPoint`
pub type ScoredPoint = qdrant::ScoredPoint;
