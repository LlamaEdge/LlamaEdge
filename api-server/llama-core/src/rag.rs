//! Define APIs for RAG operations.

use crate::{embeddings::embeddings, error::LlamaCoreError, running_mode, RunningMode};
use endpoints::{
    embeddings::{EmbeddingObject, EmbeddingsResponse, InputText},
    rag::{RagEmbeddingRequest, RagScoredPoint, RetrieveObject},
};
use qdrant::*;
use text_splitter::{MarkdownSplitter, TextSplitter};
use tiktoken_rs::cl100k_base;

/// Convert document chunks to embeddings.
///
/// # Arguments
///
/// * `embedding_request` - A reference to an `EmbeddingRequest` object.
///
/// * `qdrant_url` - URL of the Qdrant server.
///
/// * `qdrant_collection_name` - Name of the Qdrant collection to be created.
///
/// # Returns
///
/// Name of the Qdrant collection if successful.
pub async fn rag_doc_chunks_to_embeddings(
    rag_embedding_request: &RagEmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    let running_mode = running_mode()?;
    if running_mode != RunningMode::Rag {
        return Err(LlamaCoreError::Operation(format!(
            "Creating knowledge base is not supported in the {running_mode} mode.",
        )));
    }

    let embedding_request = &rag_embedding_request.embedding_request;
    let qdrant_url = rag_embedding_request.qdrant_url.as_str();
    let qdrant_collection_name = rag_embedding_request.qdrant_collection_name.as_str();

    println!("[+] Computing embeddings for document chunks...");
    if let Ok(request_str) = serde_json::to_string_pretty(&embedding_request) {
        println!("    * embedding request (json):\n\n{}", request_str);
    }

    // compute embeddings for the document
    let response = embeddings(embedding_request).await?;

    let chunks = match &embedding_request.input {
        InputText::String(text) => vec![text.clone()],
        InputText::Array(texts) => texts.clone(),
    };
    // let chunks = embedding_request.input.as_slice();
    let embeddings = response.data.as_slice();
    let dim = embeddings[0].embedding.len();

    // create a Qdrant client
    let qdrant_client = qdrant::Qdrant::new_with_url(qdrant_url.to_string());

    println!("\n[+] Creating a Qdrant collection ...");
    println!("    * Collection name: {}", qdrant_collection_name);
    println!("    * Dimension: {}", dim);

    // create a collection
    qdrant_create_collection(&qdrant_client, qdrant_collection_name, dim).await?;

    println!("\n[+] Upserting points ...");

    // create and upsert points
    qdrant_persist_embeddings(
        &qdrant_client,
        qdrant_collection_name,
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
    rag_embedding_request: &RagEmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    let running_mode = running_mode()?;
    if running_mode != RunningMode::Rag {
        return Err(LlamaCoreError::Operation(format!(
            "The RAG query is not supported in the {running_mode} mode.",
        )));
    }

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
    score_threshold: Option<f32>,
) -> Result<RetrieveObject, LlamaCoreError> {
    let running_mode = running_mode()?;
    if running_mode != RunningMode::Rag {
        return Err(LlamaCoreError::Operation(format!(
            "The context retrieval is not supported in the {running_mode} mode.",
        )));
    }

    // create a Qdrant client
    let qdrant_client = qdrant::Qdrant::new_with_url(qdrant_url.as_ref().to_string());

    // search for similar points
    let scored_points = qdrant_search_similar_points(
        &qdrant_client,
        qdrant_collection_name.as_ref(),
        query_embedding,
        limit,
        score_threshold,
    )
    .await
    .map_err(LlamaCoreError::Operation)?;

    let ro = match scored_points.is_empty() {
        true => RetrieveObject {
            points: None,
            limit,
            score_threshold: score_threshold.unwrap_or(0.0),
        },
        false => {
            let mut points: Vec<RagScoredPoint> = vec![];
            for point in scored_points.iter() {
                if let Some(payload) = &point.payload {
                    if let Some(source) = payload.get("source") {
                        points.push(RagScoredPoint {
                            source: source.to_string(),
                            score: point.score,
                        })
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
    if let Err(err) = qdrant_client
        .create_collection(collection_name.as_ref(), dim as u32)
        .await
    {
        println!("{}", err);
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

    if let Err(err) = qdrant_client
        .upsert_points(collection_name.as_ref(), points)
        .await
    {
        println!("Failed to upsert points. {}", err);
        return Err(LlamaCoreError::Operation(err.to_string()));
    }

    Ok(())
}

async fn qdrant_search_similar_points(
    qdrant_client: &qdrant::Qdrant,
    collection_name: impl AsRef<str>,
    query_vector: &[f32],
    limit: usize,
    score_threshold: Option<f32>,
) -> Result<Vec<ScoredPoint>, String> {
    match qdrant_client
        .search_points(
            collection_name.as_ref(),
            query_vector.to_vec(),
            limit as u64,
            score_threshold,
        )
        .await
    {
        Ok(search_result) => Ok(search_result),
        Err(err) => Err(err.to_string()),
    }
}

/// Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.
///
/// # Arguments
///
/// * `text` - A reference to a text.
///
/// * `ty` - Type of the text, `txt` for text content or `md` for markdown content.
///
/// * `chunk_capacity` - The max tokens each chunk contains.
///
/// # Returns
///
/// A vector of strings.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn chunk_text(
    text: impl AsRef<str>,
    ty: impl AsRef<str>,
    chunk_capacity: usize,
) -> Result<Vec<String>, LlamaCoreError> {
    if ty.as_ref().to_lowercase().as_str() != "txt" && ty.as_ref().to_lowercase().as_str() != "md" {
        return Err(LlamaCoreError::Operation(
            "Failed to upload the target file. Only files with 'txt' and 'md' extensions are supported.".to_string(),
        ));
    }

    match ty.as_ref().to_lowercase().as_str() {
        "txt" => {
            println!("[+] Chunking the text file ...");

            let tokenizer = cl100k_base().map_err(|e| LlamaCoreError::Operation(e.to_string()))?;
            let splitter = TextSplitter::new(tokenizer).with_trim_chunks(true);

            let chunks = splitter
                .chunks(text.as_ref(), chunk_capacity)
                .map(|s| s.to_string())
                .collect::<Vec<_>>();

            println!("    * Number of chunks: {}", chunks.len());

            Ok(chunks)
        },
        "md" => {
            println!("[+] Chunking the markdown file ...");

            let tokenizer = cl100k_base().map_err(|e| LlamaCoreError::Operation(e.to_string()))?;
            let splitter = MarkdownSplitter::new(tokenizer).with_trim_chunks(true);

            let chunks = splitter.chunks(text.as_ref(), chunk_capacity).map(|s| s.to_string())
            .collect::<Vec<_>>();

            println!("    * Number of chunks: {}", chunks.len());

            Ok(chunks)
        },
        _ => Err(LlamaCoreError::Operation(
            "Failed to upload the target file. Only files with 'txt' and 'md' extensions are supported.".to_string(),
        )),
    }
}
