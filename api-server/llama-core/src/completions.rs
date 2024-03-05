use crate::{
    error::{BackendError, LlamaCoreError},
    GRAPH, MAX_BUFFER_SIZE,
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

    let buffer = infer(prompt.trim()).await?;

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

/// Runs inference on the model with the given name and returns the output.
async fn infer(prompt: impl AsRef<str>) -> std::result::Result<Vec<u8>, LlamaCoreError> {
    let graph = GRAPH.get().ok_or(LlamaCoreError::Operation(
        "Fail to get the underlying value of `GRAPH`.".to_string(),
    ))?;
    let mut graph = graph.lock().map_err(|e| {
        LlamaCoreError::Operation(format!("Fail to acquire the lock of `GRAPH`. {}", e))
    })?;

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
    let mut output_buffer = vec![0u8; MAX_BUFFER_SIZE];
    let mut output_size = graph
        .get_output(0, &mut output_buffer)
        .map_err(|e| LlamaCoreError::Backend(BackendError::GetOutput(e.to_string())))?;
    output_size = std::cmp::min(MAX_BUFFER_SIZE, output_size);

    Ok(output_buffer[..output_size].to_vec())
}
