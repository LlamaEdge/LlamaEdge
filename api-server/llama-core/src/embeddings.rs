//! Define APIs for computing embeddings.

use crate::{
    error::{BackendError, LlamaCoreError},
    running_mode,
    utils::{get_output_buffer, get_token_info_by_graph},
    Graph, RunningMode, CHAT_GRAPHS, EMBEDDING_GRAPHS, OUTPUT_TENSOR,
};
use endpoints::{
    common::Usage,
    embeddings::{EmbeddingObject, EmbeddingRequest, EmbeddingsResponse, InputText},
};
use serde::{Deserialize, Serialize};

/// Compute embeddings for the given input.
///
/// # Argument
///
/// * `embedding_request` - The embedding request.
///
/// # Returns
///
/// The embeddings response.
pub async fn embeddings(
    embedding_request: &EmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    let running_mode = running_mode()?;
    if running_mode == RunningMode::Chat {
        return Err(LlamaCoreError::Operation(format!(
            "Computing embeddings is not supported in the {running_mode} mode.",
        )));
    }

    let model_name = &embedding_request.model;

    // For general embedding scenario, the embedding model is the same as the chat model.
    // For RAG scenario, the embedding model is different from the chat model.
    let embedding_graphs = match EMBEDDING_GRAPHS.get() {
        Some(embedding_graphs) => embedding_graphs,
        None => match CHAT_GRAPHS.get() {
            Some(chat_graphs) => chat_graphs,
            None => {
                return Err(LlamaCoreError::Operation(String::from(
                    "No embedding model is available.",
                )));
            }
        },
    };

    let mut embedding_graphs = embedding_graphs.lock().map_err(|e| {
        LlamaCoreError::Operation(format!(
            "Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}",
            e
        ))
    })?;

    let graph = match embedding_graphs.get_mut(model_name) {
        Some(graph) => graph,
        None => {
            return Err(LlamaCoreError::Operation(format!(
                "The model `{}` does not exist in the embedding graphs.",
                model_name
            )))
        }
    };

    // check if the `embedding` option of metadata is enabled
    if !graph.metadata.embeddings {
        graph.metadata.embeddings = true;
        graph.update_metadata()?;
    }

    // compute embeddings
    let (data, usage) = match &embedding_request.input {
        InputText::String(text) => compute_embeddings(graph, &[text.to_owned()])?,
        InputText::Array(texts) => compute_embeddings(graph, texts.as_slice())?,
    };

    // let (data, usage) = compute_embeddings(graph, &embedding_request.input)?;

    if graph.metadata.log_prompts || graph.metadata.log_enable {
        println!("[+] Embeddings computed successfully.\n");
    }

    let embedding_reponse = EmbeddingsResponse {
        object: String::from("list"),
        data,
        model: embedding_request.model.clone(),
        usage,
    };

    Ok(embedding_reponse)
}

fn compute_embeddings(
    graph: &mut Graph,
    input: &[String],
) -> Result<(Vec<EmbeddingObject>, Usage), LlamaCoreError> {
    if graph.metadata.log_prompts || graph.metadata.log_enable {
        println!("[+] Computing embeddings for {} chunks ...", input.len());
    }
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
                let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;

                // convert inference result to string
                let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
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
                let token_info = get_token_info_by_graph(graph)?;

                usage.prompt_tokens += token_info.prompt_tokens;
                usage.completion_tokens += token_info.completion_tokens;
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;

                if graph.metadata.log_prompts || graph.metadata.log_enable {
                    println!(
                        "    * chunk {} done! (prompt tokens: {})",
                        idx + 1,
                        token_info.prompt_tokens,
                    );
                }
            }
            Err(e) => {
                return Err(LlamaCoreError::Backend(BackendError::Compute(format!(
                    "Failed to compute embeddings. Reason: {}",
                    e
                ))));
            }
        }
    }

    Ok((embeddings, usage))
}

/// Get the dimension of the embedding model.
///
/// # Arguments
///
/// * `name` - The name of the embedding model. If `None`, the dimension of the first model will be returned.
///
/// # Returns
///
/// The dimension of the embedding model.
///
/// # Errors
///
/// * The model does not exist in the embedding graphs.
/// * No embedding model is available.
pub fn dimension(name: Option<&str>) -> Result<u64, LlamaCoreError> {
    let embedding_graphs =
        EMBEDDING_GRAPHS
            .get()
            .ok_or(LlamaCoreError::Operation(String::from(
                "Fail to get the underlying value of `EMBEDDING_GRAPHS`.",
            )))?;

    let embedding_graphs = embedding_graphs.lock().map_err(|e| {
        LlamaCoreError::Operation(format!(
            "Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}",
            e
        ))
    })?;

    match name {
        Some(model_name) => match embedding_graphs.get(model_name) {
            Some(graph) => Ok(graph.metadata.ctx_size),
            None => Err(LlamaCoreError::Operation(format!(
                "The model `{}` does not exist in the embedding graphs.",
                model_name
            ))),
        },
        None => {
            if !embedding_graphs.is_empty() {
                let graph = embedding_graphs
                    .values()
                    .next()
                    .ok_or(LlamaCoreError::Operation(String::from(
                        "Fail to get the underlying value of `EMBEDDING_GRAPHS`.",
                    )))?;

                Ok(graph.metadata.ctx_size)
            } else {
                Err(LlamaCoreError::Operation(String::from(
                    "No embedding model is available.",
                )))
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Embedding {
    #[serde(rename = "n_embedding")]
    len: u64,
    #[serde(rename = "embedding")]
    data: Vec<f64>,
}
