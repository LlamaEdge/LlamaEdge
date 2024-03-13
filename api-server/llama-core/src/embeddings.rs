use crate::{
    chat::get_token_info_by_graph,
    error::{BackendError, LlamaCoreError},
    Graph, EMBEDDING_GRAPHS, MAX_BUFFER_SIZE_EMBEDDING,
};
use endpoints::{
    common::Usage,
    embeddings::{EmbeddingObject, EmbeddingRequest, EmbeddingsResponse},
};
use serde::{Deserialize, Serialize};

pub async fn embeddings(
    embedding_request: &EmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    let model_name = &embedding_request.model;

    let embedding_graphs =
        EMBEDDING_GRAPHS
            .get()
            .ok_or(LlamaCoreError::Operation(String::from(
                "Fail to get the underlying value of `EMBEDDING_GRAPHS`.",
            )))?;

    let mut embedding_graphs = embedding_graphs.lock().map_err(|e| {
        LlamaCoreError::Operation(format!(
            "Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}",
            e
        ))
    })?;

    let graph = match embedding_graphs.get_mut(model_name) {
        Some(graph) => graph,
        None => {
            if !embedding_graphs.is_empty() {
                embedding_graphs
                    .values_mut()
                    .next()
                    .ok_or(LlamaCoreError::Operation(String::from(
                        "Fail to get the underlying value of `EMBEDDING_GRAPHS`.",
                    )))?
            } else {
                return Err(LlamaCoreError::Operation(format!(
                    "The model `{}` does not exist in the embedding graphs.",
                    model_name
                )));
            }
        }
    };

    // update metadata to enable the `embedding` option
    update_metadata(graph)?;

    // compute embeddings
    let (data, usage) = compute_embeddings(graph, &embedding_request.input)?;

    let embedding_reponse = EmbeddingsResponse {
        object: String::from("list"),
        data,
        model: embedding_request.model.clone(),
        usage,
    };

    Ok(embedding_reponse)
}

fn update_metadata(graph: &mut Graph) -> Result<(), LlamaCoreError> {
    let mut should_update = false;

    let mut metadata = graph.metadata.clone();

    // check if the `embedding` option is enabled
    if !metadata.embeddings {
        metadata.embeddings = true;

        if !should_update {
            should_update = true;
        }
    }

    if should_update {
        // update metadata
        let config = match serde_json::to_string(&metadata) {
            Ok(config) => config,
            Err(e) => {
                return Err(LlamaCoreError::Operation(format!(
                    "Fail to serialize metadata to a JSON string. {}",
                    e
                )));
            }
        };

        // update metadata
        if graph
            .set_input(1, wasmedge_wasi_nn::TensorType::U8, &[1], config.as_bytes())
            .is_err()
        {
            return Err(LlamaCoreError::Backend(BackendError::SetInput(
                String::from("Fail to update metadata."),
            )));
        }
    }

    Ok(())
}

fn compute_embeddings(
    graph: &mut Graph,
    // embedding_request: &EmbeddingRequest,
    input: &[String],
) -> Result<(Vec<EmbeddingObject>, Usage), LlamaCoreError> {
    // compute embeddings
    let mut embeddings: Vec<EmbeddingObject> = Vec::new();
    let mut usage = Usage::default();
    for (idx, input) in input.iter().enumerate() {
        // set input
        let tensor_data = input.as_bytes().to_vec();
        graph
            .set_input(0, wasmedge_wasi_nn::TensorType::U8, &[1], &tensor_data)
            .map_err(|e| LlamaCoreError::Backend(BackendError::SetInput(e.to_string())))?;

        match graph.compute() {
            Ok(_) => {
                // Retrieve the output.
                let mut output_buffer = vec![0u8; MAX_BUFFER_SIZE_EMBEDDING];
                let mut output_size: usize =
                    graph.get_output(0, &mut output_buffer).map_err(|e| {
                        LlamaCoreError::Operation(format!(
                            "Fail to get output tensor: {msg}",
                            msg = e
                        ))
                    })?;
                output_size = std::cmp::min(MAX_BUFFER_SIZE_EMBEDDING, output_size);

                // convert inference result to string
                let output = std::str::from_utf8(&output_buffer[..output_size]).map_err(|e| {
                    LlamaCoreError::Operation(format!(
                        "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                        e
                    ))
                })?;

                // deserialize the embedding data
                let embedding = serde_json::from_str::<Embedding>(output).map_err(|e| {
                    LlamaCoreError::Operation(format!(
                        "Failed to deserialize embedding data. {}",
                        e
                    ))
                })?;

                let embedding_object = EmbeddingObject {
                    index: idx as u64,
                    object: String::from("embedding"),
                    embedding: embedding.data,
                };

                embeddings.push(embedding_object);

                // retrieve the number of prompt and completion tokens
                let token_info = get_token_info_by_graph(&graph)?;

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

    Ok((embeddings, usage))
}

#[derive(Debug, Serialize, Deserialize)]
struct Embedding {
    #[serde(rename = "n_embedding")]
    len: u64,
    #[serde(rename = "embedding")]
    data: Vec<f64>,
}
