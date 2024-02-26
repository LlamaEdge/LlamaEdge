use crate::{
    chat::get_token_info,
    error::{BackendError, LlamaCoreError},
    GRAPH, MAX_BUFFER_SIZE_EMBEDDING, METADATA,
};
use endpoints::{
    common::Usage,
    embeddings::{EmbeddingObject, EmbeddingRequest, EmbeddingsResponse},
};
use qdrant::*;
use serde::{Deserialize, Serialize};

pub async fn embeddings(
    embedding_request: &EmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    // update metadata to enable the `embedding` option
    update_metadata()?;

    // get graph
    let graph = GRAPH.get().ok_or(LlamaCoreError::Operation(String::from(
        "Fail to get the underlying value of `GRAPH`.",
    )))?;
    let mut graph = graph.lock().map_err(|e| {
        LlamaCoreError::Operation(format!(
            "Fail to acquire the lock of `GRAPH`. {}",
            e.to_string()
        ))
    })?;

    // compute embeddings
    let mut embeddings: Vec<EmbeddingObject> = Vec::new();
    let mut usage = Usage::default();
    for (idx, input) in embedding_request.input.iter().enumerate() {
        // set input
        let tensor_data = input.as_bytes().to_vec();
        graph
            .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
            .map_err(|e| LlamaCoreError::Backend(BackendError::SetInput(e.to_string())))?;

        match graph.compute() {
            Ok(_) => {
                // Retrieve the output.
                let mut output_buffer = vec![0u8; MAX_BUFFER_SIZE_EMBEDDING];
                let mut output_size: usize =
                    graph.get_output(0, &mut output_buffer).map_err(|e| {
                        LlamaCoreError::Operation(format!(
                            "Fail to get output tensor: {msg}",
                            msg = e.to_string()
                        ))
                    })?;
                output_size = std::cmp::min(MAX_BUFFER_SIZE_EMBEDDING, output_size);

                // convert inference result to string
                let output = std::str::from_utf8(&output_buffer[..output_size]).map_err(|e| {
                    LlamaCoreError::Operation(format!(
                        "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                        e.to_string()
                    ))
                })?;

                // deserialize the embedding data
                let embedding = serde_json::from_str::<Embedding>(output).map_err(|e| {
                    LlamaCoreError::Operation(format!(
                        "Failed to deserialize embedding data. {}",
                        e.to_string()
                    ))
                })?;

                let embedding_object = EmbeddingObject {
                    index: idx as u64,
                    object: String::from("embedding"),
                    embedding: embedding.data,
                };

                embeddings.push(embedding_object);

                // retrieve the number of prompt and completion tokens
                let token_info = get_token_info(&graph).map_err(|e| {
                    LlamaCoreError::Operation(format!(
                        "Failed to get the number of prompt and completion tokens. {}",
                        e.to_string()
                    ))
                })?;

                usage.prompt_tokens += token_info.prompt_tokens;
                usage.completion_tokens += token_info.completion_tokens;
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
            }
            Err(e) => {
                return Err(LlamaCoreError::Backend(BackendError::Compute(
                    e.to_string(),
                )));
            }
        }
    }

    let embedding_reponse = EmbeddingsResponse {
        object: String::from("list"),
        data: embeddings,
        model: embedding_request.model.clone(),
        usage,
    };

    Ok(embedding_reponse)
}

fn update_metadata() -> Result<(), LlamaCoreError> {
    let mut metadata = match METADATA.get() {
        Some(metadata) => metadata.clone(),
        None => {
            return Err(LlamaCoreError::Operation(String::from(
                "Fail to get the underlying value of `METADATA`.",
            )));
        }
    };

    // enable `embedding` option
    metadata.embeddings = true;

    // update metadata
    let config = match serde_json::to_string(&metadata) {
        Ok(config) => config,
        Err(e) => {
            return Err(LlamaCoreError::Operation(format!(
                "Fail to serialize metadata to a JSON string. {}",
                e.to_string()
            )));
        }
    };

    let graph = match GRAPH.get() {
        Some(graph) => graph,
        None => {
            return Err(LlamaCoreError::Operation(String::from(
                "Fail to get the underlying value of `GRAPH`.",
            )));
        }
    };
    let mut graph = match graph.lock() {
        Ok(graph) => graph,
        Err(e) => {
            return Err(LlamaCoreError::Operation(format!(
                "Fail to acquire the lock of `GRAPH`. {}",
                e.to_string()
            )));
        }
    };

    // update metadata
    if graph
        .set_input(1, wasi_nn::TensorType::U8, &[1], config.as_bytes())
        .is_err()
    {
        return Err(LlamaCoreError::Backend(BackendError::SetInput(
            String::from("Fail to update metadata."),
        )));
    }

    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct Embedding {
    #[serde(rename = "n_embedding")]
    len: u64,
    #[serde(rename = "embedding")]
    data: Vec<f64>,
}

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
    embedding_request: &EmbeddingRequest,
    qdrant_url: impl AsRef<str>,
    qdrant_collection_name: impl AsRef<str>,
) -> Result<(), LlamaCoreError> {
    // compute embeddings for the document
    let response = embeddings(embedding_request).await?;

    let chunks = embedding_request.input.as_slice();
    let embeddings = response.data.as_slice();
    let dim = embeddings[0].embedding.len();

    // create a Qdrant client
    let qdrant_client = qdrant::Qdrant::new_with_url(qdrant_url.as_ref().to_string());

    // create a collection
    qdrant_create_collection(&qdrant_client, qdrant_collection_name.as_ref(), dim).await?;

    // create and upsert points
    qdrant_persist_embeddings(
        &qdrant_client,
        qdrant_collection_name.as_ref(),
        embeddings,
        chunks,
    )
    .await?;

    Ok(())
}

/// Convert a query to embeddings.
///
/// # Arguments
///
/// * `embedding_request` - A reference to an `EmbeddingRequest` object.
pub async fn rag_query_to_embeddings(
    embedding_request: &EmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
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
/// * `top_k` - Number of similar points to be retrieved.
pub async fn rag_retrieve_context(
    query_embedding: &[f32],
    qdrant_url: impl AsRef<str>,
    qdrant_collection_name: impl AsRef<str>,
    top_k: usize,
) -> Result<Vec<ScoredPoint>, LlamaCoreError> {
    // create a Qdrant client
    let qdrant_client = qdrant::Qdrant::new_with_url(qdrant_url.as_ref().to_string());

    // search for similar points
    let search_result = qdrant_search_similar_points(
        &qdrant_client,
        qdrant_collection_name.as_ref(),
        query_embedding,
        top_k,
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
    // let dim = points[0].vector.len();

    // // create a Qdrant client
    // let qdrant_client = qdrant::Qdrant::new();

    // // Create a collection with `dim`-dimensional vectors
    // println!("[+] Creating a collection ...");
    // // let collection_name = "my_test";
    // println!("    * Collection name: {}", collection_name.as_ref());
    // println!("    * Dimension: {}", dim);
    // if let Err(err) = qdrant_client
    //     .create_collection(collection_name.as_ref(), dim as u32)
    //     .await
    // {
    //     println!("Failed to create collection. {}", err.to_string());
    //     return Err(err.to_string());
    // }

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
