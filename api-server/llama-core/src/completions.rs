//! Define APIs for completions.

use crate::{
    error::{BackendError, LlamaCoreError},
    utils::get_output_buffer,
    Graph, CHAT_GRAPHS, OUTPUT_TENSOR,
};
use endpoints::{
    common::{FinishReason, Usage},
    completions::{CompletionChoice, CompletionObject, CompletionRequest},
};
use std::time::SystemTime;

pub async fn completions(request: &CompletionRequest) -> Result<CompletionObject, LlamaCoreError> {
    let prompt = request.prompt.join(" ");

    // ! todo: a temp solution of computing the number of tokens in prompt
    let prompt_tokens = prompt.split_whitespace().count() as u64;

    let buffer = infer(prompt.trim(), request.model.as_ref())?;

    // convert inference result to string
    let model_answer = String::from_utf8(buffer.clone()).map_err(|e| {
        LlamaCoreError::Operation(format!(
            "Failed to decode the buffer of the inference result to a utf-8 string. {}",
            e
        ))
    })?;
    let answer = model_answer.trim();

    // ! todo: a temp solution of computing the number of tokens in answer
    let completion_tokens = answer.split_whitespace().count() as u64;

    let created = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| LlamaCoreError::Operation(format!("Failed to get the current time. {}", e)))?;

    Ok(CompletionObject {
        id: uuid::Uuid::new_v4().to_string(),
        object: String::from("text_completion"),
        created: created.as_secs(),
        model: request.model.clone().unwrap_or_default(),
        choices: vec![CompletionChoice {
            index: 0,
            text: String::from(answer),
            finish_reason: FinishReason::stop,
            logprobs: None,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

fn infer(
    prompt: impl AsRef<str>,
    model_name: Option<&String>,
) -> std::result::Result<Vec<u8>, LlamaCoreError> {
    match model_name {
        Some(model_name) => {
            let chat_graphs = CHAT_GRAPHS
                .get()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `CHAT_GRAPHS`.",
                )))?;
            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `CHAT_GRAPHS`. {}",
                    e
                ))
            })?;
            match chat_graphs.get_mut(model_name) {
                Some(graph) => infer_by_graph(graph, prompt),
                None => Err(LlamaCoreError::Operation(format!(
                    "The model `{}` does not exist in the chat graphs.",
                    &model_name
                ))),
            }
        }
        None => {
            let chat_graphs = CHAT_GRAPHS
                .get()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `CHAT_GRAPHS`.",
                )))?;
            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `CHAT_GRAPHS`. {}",
                    e
                ))
            })?;

            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => infer_by_graph(graph, prompt),
                None => Err(LlamaCoreError::Operation(String::from(
                    "There is no model available in the chat graphs.",
                ))),
            }
        }
    }
}

/// Runs inference on the model with the given name and returns the output.
fn infer_by_graph(
    graph: &mut Graph,
    prompt: impl AsRef<str>,
) -> std::result::Result<Vec<u8>, LlamaCoreError> {
    // check if the `embedding` model is disabled or not
    if graph.metadata.embeddings {
        graph.metadata.embeddings = false;
        graph.update_metadata()?;
    }

    // set input
    let tensor_data = prompt.as_ref().as_bytes().to_vec();
    graph
        .set_input(0, wasmedge_wasi_nn::TensorType::U8, &[1], &tensor_data)
        .map_err(|e| LlamaCoreError::Backend(BackendError::SetInput(e.to_string())))?;

    // execute the inference
    graph
        .compute()
        .map_err(|e| LlamaCoreError::Backend(BackendError::Compute(e.to_string())))?;

    // Retrieve the output
    get_output_buffer(graph, OUTPUT_TENSOR)
}
