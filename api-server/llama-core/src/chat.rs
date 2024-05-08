//! Define APIs for chat completion.

use crate::{
    error,
    utils::{
        gen_chat_id, get_output_buffer, get_output_buffer_single, get_token_info_by_graph,
        get_token_info_by_graph_name, print_log_begin_separator, print_log_end_separator,
        set_tensor_data_u8,
    },
    Graph, Metadata, CACHED_UTF8_ENCODINGS, CHAT_GRAPHS, OUTPUT_TENSOR,
};
use chat_prompts::{
    chat::{BuildChatPrompt, ChatPrompt},
    PromptTemplateType,
};
#[cfg(feature = "https")]
use endpoints::chat::{
    ChatCompletionRequestMessage, ChatCompletionUserMessageContent, ContentPart,
};
use endpoints::{
    chat::{
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkChoiceDelta,
        ChatCompletionObject, ChatCompletionObjectChoice, ChatCompletionObjectMessage,
        ChatCompletionRequest, ChatCompletionRole,
    },
    common::{FinishReason, Usage},
};
use error::{BackendError, LlamaCoreError};
use futures::{
    future,
    stream::{self, TryStreamExt},
};
use std::{sync::Mutex, time::SystemTime};

/// Processes a chat-completion request and returns ChatCompletionChunk instances in stream.
pub async fn chat_completions_stream(
    chat_request: &mut ChatCompletionRequest,
) -> Result<impl futures::TryStream<Ok = String, Error = LlamaCoreError>, LlamaCoreError> {
    let model_name = chat_request.model.clone();

    // parse the `include_usage` option
    let include_usage = match chat_request.stream_options {
        Some(ref stream_options) => stream_options.include_usage.unwrap_or_default(),
        None => false,
    };

    // update metadata
    let mut metadata = update_metadata(chat_request).await?;

    // build prompt
    let (prompt, avaible_completion_tokens) = build_prompt(model_name.as_ref(), chat_request)
        .map_err(|e| LlamaCoreError::Operation(e.to_string()))?;

    if metadata.log_prompts {
        print_log_begin_separator("PROMPT", Some("*"), None);
        println!("\n{}", &prompt,);
        print_log_end_separator(Some("*"), None);
    }

    // update metadata n_predict
    update_n_predict(chat_request, &mut metadata, avaible_completion_tokens).await?;

    // set prompt
    set_prompt(chat_request.model.as_ref(), &prompt)?;

    let mut one_more_run_then_stop = true;
    let mut stream_state = match include_usage {
        true => StreamState::Usage,
        false => StreamState::Done,
    };
    let mut context_full_state = ContextFullState::Message;
    let mut prompt_too_long_state = PromptTooLongState::Message;

    let stream = stream::repeat_with(move || {
        // get graph
        match &model_name {
            Some(model_name) => {
                let chat_graphs =
                    CHAT_GRAPHS
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
                    Some(graph) => {
                        // compute
                        match graph.compute_single() {
                            Ok(_) => {
                                // Retrieve the output
                                let output_buffer = get_output_buffer_single(graph, OUTPUT_TENSOR)?;

                                // decode the output buffer to a utf8 string
                                let output = match String::from_utf8(output_buffer.clone())
                                {
                                    Ok(token) => token,
                                    Err(_) => {
                                        let mutex = CACHED_UTF8_ENCODINGS.get_or_init(|| Mutex::new(Vec::new()));
                                        let mut cached_encodings = mutex.lock().map_err(|e| {
                                            LlamaCoreError::Operation(format!(
                                                "Fail to acquire the lock of `UTF8_ENCODINGS`. {}",
                                                e
                                            ))
                                        })?;

                                        // cache the bytes for future decoding
                                        cached_encodings.extend_from_slice(&output_buffer[..]);

                                        match String::from_utf8(cached_encodings.to_vec()) {
                                            Ok(token) => {
                                                // clear encodings
                                                cached_encodings.clear();

                                                token
                                            }
                                            Err(_) => {
                                                // ! This is a temp check. In case, infinite cached encodings happen.
                                                if cached_encodings.len() > 3 {
                                                    return Err(LlamaCoreError::Operation(String::from(
                                                        "The length of the invalid utf8 bytes exceed 3.",
                                                    )));
                                                }

                                                String::new()
                                            }
                                        }
                                    }
                                };

                                let created = SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map_err(|e| {
                                        LlamaCoreError::Operation(format!(
                                            "Failed to get the current time. {}",
                                            e
                                        ))
                                    })?;

                                let chat_completion_chunk = ChatCompletionChunk {
                                    id: gen_chat_id(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: created.as_secs(),
                                    model: graph.name().to_owned(),
                                    system_fingerprint: "fp_44709d6fcb".to_string(),
                                    choices: vec![ChatCompletionChunkChoice {
                                        index: 0,
                                        delta: ChatCompletionChunkChoiceDelta {
                                            role: Some(ChatCompletionRole::Assistant),
                                            content: Some(output),
                                            function_call: None,
                                            tool_calls: None,
                                        },
                                        logprobs: None,
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };

                                // serialize chat completion chunk
                                let chunk_str =
                                serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                    LlamaCoreError::Operation(format!(
                                        "Failed to serialize chat completion chunk. {}",
                                        e
                                    ))
                                })?;

                                Ok(format!("data: {}\n\n", chunk_str))
                            }
                            Err(wasmedge_wasi_nn::Error::BackendError(
                                wasmedge_wasi_nn::BackendError::EndOfSequence,
                            )) => {
                                match stream_state {
                                    StreamState::Usage => {
                                        stream_state = StreamState::Done;

                                        // retrieve the number of prompt and completion tokens
                                        let token_info = get_token_info_by_graph(graph)?;

                                        let usage = Some(Usage {
                                            prompt_tokens: token_info.prompt_tokens,
                                            completion_tokens: token_info.completion_tokens,
                                            total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
                                        });

                                        let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            LlamaCoreError::Operation(format!(
                                                "Failed to get the current time. {}",
                                                e
                                            ))
                                        })?;

                                        let chat_completion_chunk = ChatCompletionChunk {
                                            id: gen_chat_id(),
                                            object: "chat.completion.chunk".to_string(),
                                            created: created.as_secs(),
                                            model: graph.name().to_owned(),
                                            system_fingerprint: "fp_44709d6fcb".to_string(),
                                            choices: vec![],
                                            usage,
                                        };

                                        // serialize chat completion chunk
                                        let chunk_str =
                                        serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                            LlamaCoreError::Operation(format!(
                                                "Failed to serialize chat completion chunk. {}",
                                                e
                                            ))
                                        })?;

                                        Ok(format!("data: {}\n\n", chunk_str))
                                    }
                                    StreamState::Done => {
                                        stream_state = StreamState::EndOfSequence;

                                        Ok("data: [DONE]\n\n".to_string())
                                    }
                                    StreamState::EndOfSequence => {
                                        // clear context
                                        if let Err(e) = graph.finish_single() {
                                            return Err(LlamaCoreError::Backend(BackendError::FinishSingle(
                                                e.to_string(),
                                            )));
                                        }

                                        Ok("[GGML] End of sequence".to_string())
                                    }
                                }
                            }
                            Err(wasmedge_wasi_nn::Error::BackendError(
                                wasmedge_wasi_nn::BackendError::ContextFull,
                            )) => {
                                match context_full_state {
                                    ContextFullState::Message => {
                                        match include_usage {
                                            true => context_full_state = ContextFullState::Usage,
                                            false => context_full_state = ContextFullState::Done,
                                        }

                                        let created = SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .map_err(|e| {
                                                LlamaCoreError::Operation(format!(
                                                    "Failed to get the current time. {}",
                                                    e
                                                ))
                                            })?;

                                        let chat_completion_chunk = ChatCompletionChunk {
                                            id: gen_chat_id(),
                                            object: "chat.completion.chunk".to_string(),
                                            created: created.as_secs(),
                                            model: graph.name().to_owned(),
                                            system_fingerprint: "fp_44709d6fcb".to_string(),
                                            choices: vec![ChatCompletionChunkChoice {
                                                index: 0,
                                                delta: ChatCompletionChunkChoiceDelta {
                                                    role: Some(ChatCompletionRole::Assistant),
                                                    content: Some("<|WASMEDGE-GGML-CONTEXT-FULL|>".to_string()),
                                                    function_call: None,
                                                    tool_calls: None,
                                                },
                                                logprobs: None,
                                                finish_reason: Some(FinishReason::length),
                                            }],
                                            usage: None,
                                        };

                                        // serialize chat completion chunk
                                        let chunk_str =
                                            serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                                LlamaCoreError::Operation(format!(
                                                    "Failed to serialize chat completion chunk. {}",
                                                    e
                                                ))
                                            })?;

                                        Ok(format!("data: {}\n\n", chunk_str))
                                    }
                                    ContextFullState::Usage => {
                                        context_full_state = ContextFullState::Done;

                                        // retrieve the number of prompt and completion tokens
                                        let token_info = get_token_info_by_graph(graph)?;

                                        let usage = Some(Usage {
                                            prompt_tokens: token_info.prompt_tokens,
                                            completion_tokens: token_info.completion_tokens,
                                            total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
                                        });

                                        let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            LlamaCoreError::Operation(format!(
                                                "Failed to get the current time. {}",
                                                e
                                            ))
                                        })?;

                                        let chat_completion_chunk = ChatCompletionChunk {
                                            id: gen_chat_id(),
                                            object: "chat.completion.chunk".to_string(),
                                            created: created.as_secs(),
                                            model: graph.name().to_owned(),
                                            system_fingerprint: "fp_44709d6fcb".to_string(),
                                            choices: vec![],
                                            usage,
                                        };

                                        // serialize chat completion chunk
                                        let chunk_str =
                                        serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                            LlamaCoreError::Operation(format!(
                                                "Failed to serialize chat completion chunk. {}",
                                                e
                                            ))
                                        })?;

                                        Ok(format!("data: {}\n\n", chunk_str))

                                    }
                                    ContextFullState::Done => {
                                        context_full_state = ContextFullState::EndOfSequence;

                                        Ok("data: [DONE]\n\n".to_string())
                                    }
                                    ContextFullState::EndOfSequence => {
                                        // clear context
                                        if let Err(e) = graph.finish_single() {
                                            return Err(LlamaCoreError::Backend(BackendError::FinishSingle(
                                                e.to_string(),
                                            )));
                                        }

                                        Ok("[GGML] End of sequence".to_string())
                                    }
                                }
                            }
                            Err(wasmedge_wasi_nn::Error::BackendError(
                                wasmedge_wasi_nn::BackendError::PromptTooLong,
                            )) => {
                                match prompt_too_long_state {
                                    PromptTooLongState::Message => {
                                        match include_usage {
                                            true => prompt_too_long_state = PromptTooLongState::Usage,
                                            false => prompt_too_long_state = PromptTooLongState::Done,
                                        }

                                        let created = SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .map_err(|e| {
                                                LlamaCoreError::Operation(format!(
                                                    "Failed to get the current time. {}",
                                                    e
                                                ))
                                            })?;

                                        let chat_completion_chunk = ChatCompletionChunk {
                                            id: gen_chat_id(),
                                            object: "chat.completion.chunk".to_string(),
                                            created: created.as_secs(),
                                            model: graph.name().to_owned(),
                                            system_fingerprint: "fp_44709d6fcb".to_string(),
                                            choices: vec![ChatCompletionChunkChoice {
                                                index: 0,
                                                delta: ChatCompletionChunkChoiceDelta {
                                                    role: Some(ChatCompletionRole::Assistant),
                                                    content: None,
                                                    function_call: None,
                                                    tool_calls: None,
                                                },
                                                logprobs: None,
                                                finish_reason: Some(FinishReason::length),
                                            }],
                                            usage: None,
                                        };

                                        // serialize chat completion chunk
                                        let chunk_str =
                                        serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                            LlamaCoreError::Operation(format!(
                                                "Failed to serialize chat completion chunk. {}",
                                                e
                                            ))
                                        })?;

                                        Ok(format!("data: {}\n\n", chunk_str))
                                    }
                                    PromptTooLongState::Usage => {
                                        prompt_too_long_state = PromptTooLongState::Done;

                                        // retrieve the number of prompt and completion tokens
                                        let token_info = get_token_info_by_graph(graph)?;

                                        let usage = Some(Usage {
                                            prompt_tokens: token_info.prompt_tokens,
                                            completion_tokens: token_info.completion_tokens,
                                            total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
                                        });

                                        let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            LlamaCoreError::Operation(format!(
                                                "Failed to get the current time. {}",
                                                e
                                            ))
                                        })?;

                                        let chat_completion_chunk = ChatCompletionChunk {
                                            id: gen_chat_id(),
                                            object: "chat.completion.chunk".to_string(),
                                            created: created.as_secs(),
                                            model: graph.name().to_owned(),
                                            system_fingerprint: "fp_44709d6fcb".to_string(),
                                            choices: vec![],
                                            usage,
                                        };

                                        // serialize chat completion chunk
                                        let chunk_str =
                                        serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                            LlamaCoreError::Operation(format!(
                                                "Failed to serialize chat completion chunk. {}",
                                                e
                                            ))
                                        })?;

                                        Ok(format!("data: {}\n\n", chunk_str))
                                    }
                                    PromptTooLongState::Done => {
                                        prompt_too_long_state = PromptTooLongState::EndOfSequence;

                                        Ok("data: [DONE]\n\n".to_string())
                                    }
                                    PromptTooLongState::EndOfSequence => {
                                        // clear context
                                        if let Err(e) = graph.finish_single() {
                                            return Err(LlamaCoreError::Backend(BackendError::FinishSingle(
                                                e.to_string(),
                                            )));
                                        }

                                        Ok("[GGML] End of sequence".to_string())
                                    }
                                }
                            }
                            Err(e) => {
                                // clear context
                                if let Err(e) = graph.finish_single() {
                                    println!("Error: {:?}", &e);
                                    return Err(LlamaCoreError::Backend(BackendError::FinishSingle(
                                        e.to_string(),
                                    )));
                                }

                                println!("Error: {:?}", &e);
                                Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                                    e.to_string(),
                                )))
                            }
                        }
                    }
                    None => {
                        Err(LlamaCoreError::Operation(format!(
                            "The model `{}` does not exist in the chat graphs.",
                            &model_name
                        )))
                    }
                }
            }
            None => {
                let chat_graphs =
                    CHAT_GRAPHS
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
                    Some((_, graph)) => {
                        // compute
                        match graph.compute_single() {
                            Ok(_) => {
                                // Retrieve the output
                                let output_buffer = get_output_buffer_single(graph, OUTPUT_TENSOR)?;
                                // decode the output buffer to a utf8 string
                                let output = match String::from_utf8(output_buffer.clone())
                                {
                                    Ok(token) => token,
                                    Err(_) => {
                                        let mutex = CACHED_UTF8_ENCODINGS.get_or_init(|| Mutex::new(Vec::new()));
                                        let mut cached_encodings = mutex.lock().map_err(|e| {
                                            LlamaCoreError::Operation(format!(
                                                "Fail to acquire the lock of `UTF8_ENCODINGS`. {}",
                                                e
                                            ))
                                        })?;

                                        cached_encodings.extend_from_slice(&output_buffer[..]);

                                        match String::from_utf8(cached_encodings.to_vec()) {
                                            Ok(token) => {
                                                // clear encodings
                                                cached_encodings.clear();

                                                token
                                            }
                                            Err(_) => {
                                                // ! This is a temp check. In case, infinite cached encodings happen.
                                                if cached_encodings.len() > 3 {
                                                    return Err(LlamaCoreError::Operation(String::from(
                                                        "The length of the invalid utf8 bytes exceed 3.",
                                                    )));
                                                }

                                                String::new()
                                            }
                                        }
                                    }
                                };

                                // ! debug
                                // retrieve the number of prompt and completion tokens
                                let token_info = get_token_info_by_graph(graph)?;
                                println!("[DEBUG] token_info: {:?}", token_info);

                                let created = SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map_err(|e| {
                                        LlamaCoreError::Operation(format!(
                                            "Failed to get the current time. {}",
                                            e
                                        ))
                                    })?;

                                let chat_completion_chunk = ChatCompletionChunk {
                                    id: gen_chat_id(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: created.as_secs(),
                                    model: graph.name().to_owned(),
                                    system_fingerprint: "fp_44709d6fcb".to_string(),
                                    choices: vec![ChatCompletionChunkChoice {
                                        index: 0,
                                        delta: ChatCompletionChunkChoiceDelta {
                                            role: Some(ChatCompletionRole::Assistant),
                                            content: Some(output),
                                            function_call: None,
                                            tool_calls: None,
                                        },
                                        logprobs: None,
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };

                                // serialize chat completion chunk
                                let chunk_str =
                                serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                    LlamaCoreError::Operation(format!(
                                        "Failed to serialize chat completion chunk. {}",
                                        e
                                    ))
                                })?;

                                Ok(format!("data: {}\n\n", chunk_str))
                            }
                            Err(wasmedge_wasi_nn::Error::BackendError(
                                wasmedge_wasi_nn::BackendError::EndOfSequence,
                            )) => {
                                match one_more_run_then_stop {
                                    true => {
                                        one_more_run_then_stop = false;

                                        Ok("data: [DONE]\n\n".to_string())
                                    }
                                    false => {
                                        // clear context
                                        if let Err(e) = graph.finish_single() {
                                            return Err(LlamaCoreError::Backend(BackendError::FinishSingle(
                                                e.to_string(),
                                            )));
                                        }

                                        Ok("[GGML] End of sequence".to_string())
                                    }
                                }
                            }
                            Err(wasmedge_wasi_nn::Error::BackendError(
                                wasmedge_wasi_nn::BackendError::ContextFull,
                            )) => {
                                match one_more_run_then_stop {
                                    true => {
                                        let created = SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .map_err(|e| {
                                                LlamaCoreError::Operation(format!(
                                                    "Failed to get the current time. {}",
                                                    e
                                                ))
                                            })?;

                                        let chat_completion_chunk = ChatCompletionChunk {
                                            id: gen_chat_id(),
                                            object: "chat.completion.chunk".to_string(),
                                            created: created.as_secs(),
                                            model: graph.name().to_owned(),
                                            system_fingerprint: "fp_44709d6fcb".to_string(),
                                            choices: vec![ChatCompletionChunkChoice {
                                                index: 0,
                                                delta: ChatCompletionChunkChoiceDelta {
                                                    role: Some(ChatCompletionRole::Assistant),
                                                    content: Some("<|WASMEDGE-GGML-CONTEXT-FULL|>".to_string()),
                                                    function_call: None,
                                                    tool_calls: None,
                                                },
                                                logprobs: None,
                                                finish_reason: Some(FinishReason::length),
                                            }],
                                            usage: None,
                                        };

                                        one_more_run_then_stop = false;

                                        // serialize chat completion chunk
                                        let chunk_str =
                                            serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                                LlamaCoreError::Operation(format!(
                                                    "Failed to serialize chat completion chunk. {}",
                                                    e
                                                ))
                                            })?;

                                        Ok(format!("data: {}\n\n", chunk_str))
                                    }
                                    false => {
                                        // clear context
                                        if let Err(e) = graph.finish_single() {
                                            return Err(LlamaCoreError::Backend(BackendError::FinishSingle(
                                                e.to_string(),
                                            )));
                                        }

                                        Ok("[GGML] End of sequence".to_string())
                                    }
                                }
                            }
                            Err(wasmedge_wasi_nn::Error::BackendError(
                                wasmedge_wasi_nn::BackendError::PromptTooLong,
                            )) => {
                                match one_more_run_then_stop {
                                    true => {
                                        let created = SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .map_err(|e| {
                                                LlamaCoreError::Operation(format!(
                                                    "Failed to get the current time. {}",
                                                    e
                                                ))
                                            })?;

                                        let chat_completion_chunk = ChatCompletionChunk {
                                            id: gen_chat_id(),
                                            object: "chat.completion.chunk".to_string(),
                                            created: created.as_secs(),
                                            model: graph.name().to_owned(),
                                            system_fingerprint: "fp_44709d6fcb".to_string(),
                                            choices: vec![ChatCompletionChunkChoice {
                                                index: 0,
                                                delta: ChatCompletionChunkChoiceDelta {
                                                    role: Some(ChatCompletionRole::Assistant),
                                                    content: None,
                                                    function_call: None,
                                                    tool_calls: None,
                                                },
                                                logprobs: None,
                                                finish_reason: Some(FinishReason::length),
                                            }],
                                            usage: None,
                                        };

                                        one_more_run_then_stop = false;

                                        // serialize chat completion chunk
                                        let chunk_str =
                                        serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                            LlamaCoreError::Operation(format!(
                                                "Failed to serialize chat completion chunk. {}",
                                                e
                                            ))
                                        })?;

                                        Ok(format!("data: {}\n\n", chunk_str))
                                    }
                                    false => {
                                        // clear context
                                        if let Err(e) = graph.finish_single() {
                                            return Err(LlamaCoreError::Backend(BackendError::FinishSingle(
                                                e.to_string(),
                                            )));
                                        }

                                        Ok("[GGML] End of sequence".to_string())
                                    }
                                }
                            }
                            Err(e) => {
                                // clear context
                                if let Err(e) = graph.finish_single() {
                                    println!("Error: {:?}", &e);
                                    return Err(LlamaCoreError::Backend(BackendError::FinishSingle(
                                        e.to_string(),
                                    )));
                                }

                                println!("Error: {:?}", &e);
                                Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                                    e.to_string(),
                                )))
                            }
                        }
                    }
                    None => {
                        Err(LlamaCoreError::Operation(String::from(
                            "There is no model available in the chat graphs.",
                        )))
                    }
                }
            }
        }
    })
    .try_take_while(|x| future::ready(Ok(x != "[GGML] End of sequence" && !x.is_empty())));

    Ok(stream)
}

/// Processes a chat-completion request and returns a ChatCompletionObject instance.
pub async fn chat_completions(
    chat_request: &mut ChatCompletionRequest,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    let model_name = chat_request.model.clone();

    // update metadata
    let mut metadata = update_metadata(chat_request).await?;

    // build prompt
    let (prompt, avaible_completion_tokens) = build_prompt(model_name.as_ref(), chat_request)
        .map_err(|e| LlamaCoreError::Operation(format!("Failed to build prompt. {}", e)))?;

    if metadata.log_prompts {
        print_log_begin_separator("PROMPT", Some("*"), None);
        println!("\n{}", &prompt,);
        print_log_end_separator(Some("*"), None);
    }

    // update metadata n_predict
    update_n_predict(chat_request, &mut metadata, avaible_completion_tokens).await?;

    // set prompt
    set_prompt(model_name.as_ref(), &prompt)?;

    // compute
    compute(model_name.as_ref())
}

fn compute(model_name: Option<&String>) -> Result<ChatCompletionObject, LlamaCoreError> {
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
                Some(graph) => compute_by_graph(graph),
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
                Some((_, graph)) => compute_by_graph(graph),
                None => Err(LlamaCoreError::Operation(String::from(
                    "There is no model available in the chat graphs.",
                ))),
            }
        }
    }
}

fn compute_by_graph(graph: &mut Graph) -> Result<ChatCompletionObject, LlamaCoreError> {
    match graph.compute() {
        Ok(_) => {
            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                ))
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                LlamaCoreError::Operation(format!("Failed to post-process the output. {}", e))
            })?;

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;
            if graph.metadata.log_prompts {
                print_log_begin_separator("PROMPT (Tokens)", Some("*"), None);
                println!(
                    "\nprompt tokens: {}, completion_tokens: {}",
                    token_info.prompt_tokens, token_info.completion_tokens
                );
                print_log_end_separator(Some("*"), None);
            }

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    LlamaCoreError::Operation(format!("Failed to get the current time. {}", e))
                })?;

            // create ChatCompletionResponse
            Ok(ChatCompletionObject {
                id: gen_chat_id(),
                object: String::from("chat.completion"),
                created: created.as_secs(),
                model: graph.name().to_owned(),
                choices: vec![ChatCompletionObjectChoice {
                    index: 0,
                    message: ChatCompletionObjectMessage {
                        role: ChatCompletionRole::Assistant,
                        content: message,
                        function_call: None,
                    },
                    finish_reason: FinishReason::stop,
                }],
                usage: Usage {
                    prompt_tokens: token_info.prompt_tokens,
                    completion_tokens: token_info.completion_tokens,
                    total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
                },
            })
        }
        Err(wasmedge_wasi_nn::Error::BackendError(wasmedge_wasi_nn::BackendError::ContextFull)) => {
            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                ))
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                LlamaCoreError::Operation(format!("Failed to post-process the output. {}", e))
            })?;

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;
            if graph.metadata.log_prompts {
                print_log_begin_separator("PROMPT (Tokens)", Some("*"), None);
                println!(
                    "\nprompt tokens: {}, completion_tokens: {}",
                    token_info.prompt_tokens, token_info.completion_tokens
                );
                print_log_end_separator(Some("*"), None);
            }

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    LlamaCoreError::Operation(format!("Failed to get the current time. {}", e))
                })?;

            // create ChatCompletionResponse
            Ok(ChatCompletionObject {
                id: gen_chat_id(),
                object: String::from("chat.completion"),
                created: created.as_secs(),
                model: graph.name().to_owned(),
                choices: vec![ChatCompletionObjectChoice {
                    index: 0,
                    message: ChatCompletionObjectMessage {
                        role: ChatCompletionRole::Assistant,
                        content: message,
                        function_call: None,
                    },
                    finish_reason: FinishReason::length,
                }],
                usage: Usage {
                    prompt_tokens: token_info.prompt_tokens,
                    completion_tokens: token_info.completion_tokens,
                    total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
                },
            })
        }
        Err(wasmedge_wasi_nn::Error::BackendError(
            wasmedge_wasi_nn::BackendError::PromptTooLong,
        )) => {
            println!("\n\n[WARNING] The prompt is too long. Please reduce the length of your input and try again.\n");

            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                ))
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                LlamaCoreError::Operation(format!("Failed to post-process the output. {}", e))
            })?;

            // retrieve the number of prompt and completion token
            let token_info = get_token_info_by_graph(graph)?;
            if graph.metadata.log_prompts {
                print_log_begin_separator("PROMPT (Tokens)", Some("*"), None);
                println!(
                    "\nprompt tokens: {}, completion_tokens: {}",
                    token_info.prompt_tokens, token_info.completion_tokens
                );
                print_log_end_separator(Some("*"), None);
            }

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    LlamaCoreError::Operation(format!("Failed to get the current time. {}", e))
                })?;

            // create ChatCompletionResponse
            Ok(ChatCompletionObject {
                id: gen_chat_id(),
                object: String::from("chat.completion"),
                created: created.as_secs(),
                model: graph.name().to_owned(),
                choices: vec![ChatCompletionObjectChoice {
                    index: 0,
                    message: ChatCompletionObjectMessage {
                        role: ChatCompletionRole::Assistant,
                        content: message,
                        function_call: None,
                    },
                    finish_reason: FinishReason::length,
                }],
                usage: Usage {
                    prompt_tokens: token_info.prompt_tokens,
                    completion_tokens: token_info.completion_tokens,
                    total_tokens: token_info.completion_tokens + token_info.completion_tokens,
                },
            })
        }
        Err(e) => Err(LlamaCoreError::Backend(BackendError::Compute(
            e.to_string(),
        ))),
    }
}

async fn update_metadata(chat_request: &ChatCompletionRequest) -> Result<Metadata, LlamaCoreError> {
    let mut should_update = false;
    let mut metadata = get_metadata(chat_request.model.as_ref())?;

    // check if necessary to update `image`
    #[cfg(feature = "https")]
    if let Some(ChatCompletionRequestMessage::User(user_message)) = chat_request.messages.last() {
        if let ChatCompletionUserMessageContent::Parts(parts) = user_message.content() {
            for part in parts {
                if let ContentPart::Image(image) = part {
                    let image = image.image();

                    if image.is_url() {
                        // update metadata image
                        let img = download_image(&image.url).await?;

                        metadata.image = Some(img);

                        if !should_update {
                            should_update = true;
                        }

                        // todo: now only support a single image
                        break;
                    }
                }
            }
        }
    }

    // check if necessary to update temperature
    if let Some(temp) = chat_request.temperature {
        if metadata.temperature != temp {
            // update temperature
            metadata.temperature = temp;

            if !should_update {
                should_update = true;
            }
        }
    }

    // check if necessary to update top_p
    if let Some(top_p) = chat_request.top_p {
        if metadata.top_p != top_p {
            // update top_p
            metadata.top_p = top_p;

            if !should_update {
                should_update = true;
            }
        }
    }

    // check if necessary to update frequency_penalty
    if let Some(frequency_penalty) = chat_request.frequency_penalty {
        if metadata.frequency_penalty != frequency_penalty {
            // update frequency_penalty
            metadata.frequency_penalty = frequency_penalty;

            if !should_update {
                should_update = true;
            }
        }
    }

    // check if necessary to update presence_penalty
    if let Some(presence_penalty) = chat_request.presence_penalty {
        if metadata.presence_penalty != presence_penalty {
            // update presence_penalty
            metadata.presence_penalty = presence_penalty;

            if !should_update {
                should_update = true;
            }
        }
    }

    // check if the `embedding` option is disabled
    if metadata.embeddings {
        metadata.embeddings = false;

        if !should_update {
            should_update = true;
        }
    }

    if should_update {
        // update the target graph with the new metadata
        set_metadata(chat_request.model.as_ref(), &metadata)?;
    }

    Ok(metadata)
}

async fn update_n_predict(
    chat_request: &ChatCompletionRequest,
    metadata: &mut Metadata,
    available_completion_tokens: u64,
) -> Result<(), LlamaCoreError> {
    let mut should_update = false;

    // check if necessary to update n_predict with max_tokens
    if let Some(max_tokens) = chat_request.max_tokens {
        let max_completion_tokens = match available_completion_tokens < max_tokens {
            true => available_completion_tokens,
            false => max_tokens,
        };

        // update n_predict
        metadata.n_predict = max_completion_tokens;

        if !should_update {
            should_update = true;
        }
    } else if metadata.n_predict > available_completion_tokens {
        // update n_predict
        metadata.n_predict = available_completion_tokens;

        if !should_update {
            should_update = true;
        }
    }

    if should_update {
        // update the target graph with the new metadata
        set_metadata(chat_request.model.as_ref(), metadata)?;
    }

    Ok(())
}

fn post_process(
    output: impl AsRef<str>,
    template_ty: &PromptTemplateType,
) -> Result<String, String> {
    let output = if *template_ty == PromptTemplateType::Baichuan2 {
        if output.as_ref().contains("用户:") {
            output.as_ref().trim_end_matches("用户:").trim().to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::OpenChat {
        if output.as_ref().contains("<|end_of_turn|>") {
            output
                .as_ref()
                .trim_end_matches("<|end_of_turn|>")
                .trim()
                .to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::ChatML {
        if output.as_ref().contains("<|im_start|>") && output.as_ref().contains("<|im_end|>") {
            let idx_start = output.as_ref().find("<|im_start|>").unwrap();
            let idx_end = output.as_ref().find("<|im_end|>").unwrap();

            match idx_start <= idx_end {
                true => output.as_ref().split("<|im_start|>").collect::<Vec<_>>()[0]
                    .trim()
                    .to_owned(),
                false => output.as_ref().split("<|im_end|>").collect::<Vec<_>>()[0]
                    .trim()
                    .to_owned(),
            }
        } else if output.as_ref().contains("<|im_start|>") {
            output.as_ref().split("<|im_start|>").collect::<Vec<_>>()[0]
                .trim()
                .to_owned()
        } else if output.as_ref().contains("<|im_end|>") {
            output.as_ref().split("<|im_end|>").collect::<Vec<_>>()[0]
                .trim()
                .to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::Zephyr
        || *template_ty == PromptTemplateType::MistralLite
    {
        if output.as_ref().contains("</s><") {
            output.as_ref().trim_end_matches("</s><").trim().to_owned()
        } else if output.as_ref().contains("</s>") {
            output
                .as_ref()
                .strip_suffix("</s>")
                .unwrap()
                .trim()
                .to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::DeepseekChat {
        if output.as_ref().contains("<|end_of_sentence|>") {
            output
                .as_ref()
                .trim_end_matches("<|end_of_sentence|>")
                .trim()
                .replace("<|end_of_sentence|>", " ")
                .trim()
                .to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::HumanAssistant {
        if output.as_ref().contains("Human:") {
            output.as_ref().trim_end_matches("Human:").trim().to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::SolarInstruct {
        let s = output.as_ref().trim();

        if s.starts_with("### Answer") {
            let s = s.trim_start_matches("###").trim();

            if s.starts_with("Answer:\n") {
                s.replace("Answer:\n", "Answer: ")
            } else {
                s.to_owned()
            }
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Llama2Chat {
        let s = output.as_ref().trim();
        if s.ends_with("</s>") {
            s.trim_end_matches("</s>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Llama3Chat {
        let s = output.as_ref().trim();
        if s.ends_with("<|eot_id|>") {
            s.trim_end_matches("<|eot_id|>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Phi3Chat {
        let s = output.as_ref().trim();
        if s.ends_with("<|end|>") {
            s.trim_end_matches("<|end|>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else {
        output.as_ref().trim().to_owned()
    };

    Ok(output)
}

fn build_prompt(
    model_name: Option<&String>,
    // template: &ChatPrompt,
    chat_request: &mut ChatCompletionRequest,
) -> Result<(String, u64), LlamaCoreError> {
    let metadata = get_metadata(model_name)?;
    let ctx_size = metadata.ctx_size as u64;
    let chat_prompt = ChatPrompt::from(metadata.prompt_template);

    // compute max prompt tokens
    let max_prompt_tokens = ctx_size * 4 / 5;

    loop {
        // build prompt
        let prompt = match chat_prompt.build(&mut chat_request.messages) {
            Ok(prompt) => prompt,
            Err(e) => {
                return Err(LlamaCoreError::Operation(format!(
                    "Fail to build chat prompts: {msg}",
                    msg = e
                )))
            }
        };

        // set prompt
        set_prompt(model_name, &prompt)?;

        // Retrieve the number of prompt tokens.
        let token_info = get_token_info_by_graph_name(model_name)?;

        match token_info.prompt_tokens > max_prompt_tokens {
            true => {
                match chat_request.messages[0].role() {
                    ChatCompletionRole::System => {
                        if chat_request.messages.len() >= 4 {
                            if chat_request.messages[1].role() == ChatCompletionRole::User {
                                chat_request.messages.remove(1);
                            }
                            if chat_request.messages[1].role() == ChatCompletionRole::Assistant {
                                chat_request.messages.remove(1);
                            }
                        } else if chat_request.messages.len() == 3
                            && chat_request.messages[1].role() == ChatCompletionRole::User
                        {
                            chat_request.messages.remove(1);
                        } else {
                            return Ok((prompt, ctx_size - max_prompt_tokens));
                        }
                    }
                    ChatCompletionRole::User => {
                        if chat_request.messages.len() >= 3 {
                            if chat_request.messages[0].role() == ChatCompletionRole::User {
                                chat_request.messages.remove(0);
                            }
                            if chat_request.messages[0].role() == ChatCompletionRole::Assistant {
                                chat_request.messages.remove(0);
                            }
                        } else if chat_request.messages.len() == 2
                            && chat_request.messages[0].role() == ChatCompletionRole::User
                        {
                            chat_request.messages.remove(0);
                        } else {
                            return Ok((prompt, ctx_size - max_prompt_tokens));
                        }
                    }
                    _ => panic!("Found a unsupported chat message role!"),
                }

                continue;
            }
            false => return Ok((prompt, ctx_size - max_prompt_tokens)),
        }
    }
}

/// Downloads an image from the given URL and returns the file name.
#[cfg(feature = "https")]
async fn download_image(image_url: impl AsRef<str>) -> Result<String, LlamaCoreError> {
    let image_url = image_url.as_ref();
    let url =
        reqwest::Url::parse(image_url).map_err(|e| LlamaCoreError::Operation(e.to_string()))?;
    let response = reqwest::get(url)
        .await
        .map_err(|e| LlamaCoreError::Operation(e.to_string()))?;

    let fname = response
        .url()
        .path_segments()
        .and_then(|segments| segments.last())
        .and_then(|name| if name.is_empty() { None } else { Some(name) })
        .ok_or(LlamaCoreError::Operation(format!(
            "Fail to get the file name: {}",
            image_url
        )))?
        .to_string();

    let mut dest =
        std::fs::File::create(&fname).map_err(|e| LlamaCoreError::Operation(e.to_string()))?;

    let mut content = response.bytes_stream();
    while let Some(item) = content.try_next().await.unwrap() {
        std::io::copy(&mut item.as_ref(), &mut dest)
            .map_err(|e| LlamaCoreError::Operation(e.to_string()))?;
    }

    Ok(fname)
}

fn set_prompt(model_name: Option<&String>, prompt: impl AsRef<str>) -> Result<(), LlamaCoreError> {
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
                Some(graph) => {
                    let tensor_data = prompt.as_ref().as_bytes().to_vec();
                    set_tensor_data_u8(graph, 0, &tensor_data)
                }
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
                Some((_, graph)) => {
                    let tensor_data = prompt.as_ref().as_bytes().to_vec();
                    set_tensor_data_u8(graph, 0, &tensor_data)
                }
                None => Err(LlamaCoreError::Operation(String::from(
                    "There is no model available in the chat graphs.",
                ))),
            }
        }
    }
}

// fn set_tensor_data_u8(
//     graph: &mut Graph,
//     idx: usize,
//     tensor_data: &[u8],
// ) -> Result<(), LlamaCoreError> {
//     if graph
//         .set_input(idx, wasmedge_wasi_nn::TensorType::U8, &[1], tensor_data)
//         .is_err()
//     {
//         return Err(LlamaCoreError::Operation(String::from(
//             "Fail to set input tensor",
//         )));
//     };

//     Ok(())
// }

fn get_metadata(model_name: Option<&String>) -> Result<Metadata, LlamaCoreError> {
    match model_name {
        Some(model_name) => {
            let chat_graphs = CHAT_GRAPHS
                .get()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `CHAT_GRAPHS`.",
                )))?;
            let chat_graphs = chat_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `CHAT_GRAPHS`. {}",
                    e
                ))
            })?;
            match chat_graphs.get(model_name) {
                Some(graph) => Ok(graph.metadata.clone()),
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
            let chat_graphs = chat_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `CHAT_GRAPHS`. {}",
                    e
                ))
            })?;

            match chat_graphs.iter().next() {
                Some((_, graph)) => Ok(graph.metadata.clone()),
                None => Err(LlamaCoreError::Operation(String::from(
                    "There is no model available in the chat graphs.",
                ))),
            }
        }
    }
}

fn set_metadata(model_name: Option<&String>, metadata: &Metadata) -> Result<(), LlamaCoreError> {
    let config = match serde_json::to_string(metadata) {
        Ok(config) => config,
        Err(e) => {
            return Err(LlamaCoreError::Operation(format!(
                "Fail to serialize metadata to a JSON string. {}",
                e
            )));
        }
    };

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
                Some(graph) => {
                    // update metadata
                    set_tensor_data_u8(graph, 1, config.as_bytes())
                }
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
                Some((_, graph)) => {
                    // update metadata
                    set_tensor_data_u8(graph, 1, config.as_bytes())
                }
                None => Err(LlamaCoreError::Operation(String::from(
                    "There is no model available in the chat graphs.",
                ))),
            }
        }
    }
}

#[derive(Debug)]
enum ContextFullState {
    Message,
    Usage,
    Done,
    EndOfSequence,
}

#[derive(Debug)]
enum StreamState {
    Usage,
    Done,
    EndOfSequence,
}

#[derive(Debug)]
enum PromptTooLongState {
    Message,
    Usage,
    Done,
    EndOfSequence,
}
