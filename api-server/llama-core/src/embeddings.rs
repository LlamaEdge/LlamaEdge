use crate::{
    chat::get_token_info,
    error::{BackendError, LlamaCoreError},
    GRAPH, MAX_BUFFER_SIZE_EMBEDDING, METADATA,
};
use endpoints::{
    common::Usage,
    embeddings::{EmbeddingObject, EmbeddingRequest, EmbeddingsResponse},
};
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

                println!("token_info: {:?}", &token_info);

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
