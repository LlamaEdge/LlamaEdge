//! Define APIs for completions.

use crate::{
    error::{BackendError, LlamaCoreError},
    running_mode,
    utils::{get_output_buffer, get_token_info_by_graph},
    Graph, RunningMode, CHAT_GRAPHS, OUTPUT_TENSOR,
};
use endpoints::{
    common::{FinishReason, Usage},
    completions::{CompletionChoice, CompletionObject, CompletionRequest},
};
use std::time::SystemTime;

/// Given a prompt, the model will return one or more predicted completions along with the probabilities of alternative tokens at each position.
pub async fn completions(request: &CompletionRequest) -> Result<CompletionObject, LlamaCoreError> {
    let running_mode = running_mode()?;
    if running_mode == RunningMode::Embeddings || running_mode == RunningMode::Rag {
        return Err(LlamaCoreError::Operation(format!(
            "The completion is not supported in the {running_mode} mode.",
        )));
    }

    let prompt = request.prompt.join(" ");

    compute(prompt.trim(), request.model.as_ref())
}

fn compute(
    prompt: impl AsRef<str>,
    model_name: Option<&String>,
) -> std::result::Result<CompletionObject, LlamaCoreError> {
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
                Some(graph) => compute_by_graph(graph, prompt),
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
                Some((_, graph)) => compute_by_graph(graph, prompt),
                None => Err(LlamaCoreError::Operation(String::from(
                    "There is no model available in the chat graphs.",
                ))),
            }
        }
    }
}

/// Runs inference on the model with the given name and returns the output.
fn compute_by_graph(
    graph: &mut Graph,
    prompt: impl AsRef<str>,
) -> std::result::Result<CompletionObject, LlamaCoreError> {
    // check if the `embedding` model is disabled or not
    if graph.metadata.embeddings {
        graph.metadata.embeddings = false;
        graph.update_metadata()?;
    }

    if graph.metadata.log_prompts || graph.metadata.log_enable {
        println!("[+] Setting prompt tensor ...");
    }
    // set input
    let tensor_data = prompt.as_ref().as_bytes().to_vec();
    graph
        .set_input(0, wasmedge_wasi_nn::TensorType::U8, &[1], &tensor_data)
        .map_err(|e| LlamaCoreError::Backend(BackendError::SetInput(e.to_string())))?;

    if graph.metadata.log_prompts || graph.metadata.log_enable {
        println!("[+] Generating completion tokens ...");
    }
    // execute the inference
    graph
        .compute()
        .map_err(|e| LlamaCoreError::Backend(BackendError::Compute(e.to_string())))?;

    // Retrieve the output
    let buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;

    // convert inference result to string
    let model_answer = String::from_utf8(buffer).map_err(|e| {
        LlamaCoreError::Operation(format!(
            "Failed to decode the buffer of the inference result to a utf-8 string. {}",
            e
        ))
    })?;
    let answer = model_answer.trim();

    // retrieve the number of prompt and completion tokens
    let token_info = get_token_info_by_graph(graph)?;
    if graph.metadata.log_prompts {
        println!(
            "    * prompt tokens: {}, completion_tokens: {}",
            token_info.prompt_tokens, token_info.completion_tokens
        );
    }

    let created = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| LlamaCoreError::Operation(format!("Failed to get the current time. {}", e)))?;

    if graph.metadata.log_prompts || graph.metadata.log_enable {
        println!("[+] Completions generated successfully.\n");
    }
    Ok(CompletionObject {
        id: uuid::Uuid::new_v4().to_string(),
        object: String::from("text_completion"),
        created: created.as_secs(),
        model: graph.name().to_string(),
        choices: vec![CompletionChoice {
            index: 0,
            text: String::from(answer),
            finish_reason: FinishReason::stop,
            logprobs: None,
        }],
        usage: Usage {
            prompt_tokens: token_info.prompt_tokens,
            completion_tokens: token_info.completion_tokens,
            total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
        },
    })
}
