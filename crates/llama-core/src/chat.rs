//! Define APIs for chat completion.

use crate::{
    error,
    metadata::ggml::GgmlMetadata,
    running_mode,
    utils::{
        gen_chat_id, get_output_buffer, get_output_buffer_single, get_token_info_by_graph,
        get_token_info_by_graph_name, set_tensor_data_u8,
    },
    Graph, RunningMode, CACHED_UTF8_ENCODINGS, CHAT_GRAPHS, OUTPUT_TENSOR,
};
use chat_prompts::{
    chat::{BuildChatPrompt, ChatPrompt},
    PromptTemplateType,
};
use either::{Either, Left, Right};
use endpoints::{
    chat::{
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkChoiceDelta,
        ChatCompletionObject, ChatCompletionObjectChoice, ChatCompletionObjectMessage,
        ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionRole,
        ChatCompletionUserMessageContent, ContentPart, Function, ToolCall, ToolCallForChunk,
        ToolChoice,
    },
    common::{FinishReason, Usage},
};
use error::{BackendError, LlamaCoreError};
use futures::StreamExt;
use std::{
    collections::VecDeque,
    fs::{self, File},
    path::Path,
    pin::Pin,
    sync::Mutex,
    task::{Context, Poll},
    time::SystemTime,
};

/// Processes a chat-completion request and returns either a stream of ChatCompletionChunk instances or a ChatCompletionObject instance.
pub async fn chat(
    chat_request: &mut ChatCompletionRequest,
) -> Result<
    Either<impl futures::TryStream<Ok = String, Error = LlamaCoreError>, ChatCompletionObject>,
    LlamaCoreError,
> {
    #[cfg(feature = "logging")]
    {
        debug!(target: "stdout", "tool choice: {:?}", chat_request.tool_choice.as_ref());
        debug!(target: "stdout", "tools: {:?}", chat_request.tools.as_ref());
        debug!(target: "stdout", "stream mode: {:?}", chat_request.stream);
    }

    let model_name = chat_request.model.clone();

    let result = match chat_request.stream {
        Some(true) => match chat_stream(chat_request).await {
            Ok(stream) => Ok(Left(stream)),
            Err(e) => Err(e),
        },
        Some(false) | None => match chat_once(chat_request).await {
            Ok(chat_completion_object) => Ok(Right(chat_completion_object)),
            Err(e) => Err(e),
        },
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Reset the model metadata");

    // reset the model metadata
    reset_model_metadata(model_name.as_ref())?;

    result
}

/// Processes a chat-completion request and returns ChatCompletionChunk instances in stream.
#[deprecated(since = "0.10.0", note = "Please use the `chat` function.")]
pub async fn chat_completions_stream(
    chat_request: &mut ChatCompletionRequest,
) -> Result<impl futures::TryStream<Ok = String, Error = LlamaCoreError>, LlamaCoreError> {
    chat_stream(chat_request).await
}

/// Processes a chat-completion request and returns a ChatCompletionObject instance.
#[deprecated(since = "0.10.0", note = "Please use the `chat` function.")]
pub async fn chat_completions(
    chat_request: &mut ChatCompletionRequest,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    chat_once(chat_request).await
}

async fn chat_stream(
    chat_request: &mut ChatCompletionRequest,
) -> Result<impl futures::TryStream<Ok = String, Error = LlamaCoreError>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Process chat completion request in the stream mode");

    let running_mode = running_mode()?;
    if !running_mode.contains(RunningMode::CHAT) && !running_mode.contains(RunningMode::RAG) {
        let err_msg = "The chat completion is only supported in the chat or rag mode.";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::Operation(err_msg.to_string()));
    }

    let model_name = chat_request.model.clone();
    let id = match &chat_request.user {
        Some(id) => id.clone(),
        None => gen_chat_id(),
    };
    #[cfg(feature = "logging")]
    info!(target: "stdout", "user: {}", &id);

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Check model metadata");

    // update metadata
    let mut metadata = check_model_metadata(chat_request).await?;

    // parse the `include_usage` option
    let include_usage = match chat_request.stream_options {
        Some(ref stream_options) => stream_options.include_usage.unwrap_or_default(),
        None => metadata.include_usage,
    };
    #[cfg(feature = "logging")]
    info!(target: "stdout", "include_usage: {}", include_usage);

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Build the chat prompt");

    // build prompt
    let (prompt, avaible_completion_tokens, tool_use) =
        build_prompt(model_name.as_ref(), chat_request)?;

    #[cfg(feature = "logging")]
    {
        info!(target: "stdout", "prompt:\n{}", &prompt);
        info!(target: "stdout", "available_completion_tokens: {}", avaible_completion_tokens);
        info!(target: "stdout", "tool_use: {}", tool_use);
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Update the n_predict");

    // update metadata n_predict
    update_n_predict(chat_request, &mut metadata, avaible_completion_tokens).await?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Feed the prompt to the model");

    // set prompt
    set_prompt(chat_request.model.as_ref(), &prompt)?;

    let stream = match tool_use {
        false => ChatStream::new(model_name, id, include_usage, None),
        true => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match model_name {
                Some(model_name) => match chat_graphs.contains_key(&model_name) {
                    true => {
                        let graph = chat_graphs.get_mut(&model_name).unwrap();
                        chat_stream_by_graph(graph, id, include_usage)?
                    }
                    false => match chat_graphs.iter_mut().next() {
                        Some((_, graph)) => chat_stream_by_graph(graph, id, include_usage)?,
                        None => {
                            let err_msg = "There is no model available in the chat graphs.";

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg.into()));
                        }
                    },
                },
                None => match chat_graphs.iter_mut().next() {
                    Some((_, graph)) => chat_stream_by_graph(graph, id, include_usage)?,
                    None => {
                        let err_msg = "There is no model available in the chat graphs.";

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        return Err(LlamaCoreError::Operation(err_msg.into()));
                    }
                },
            }
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "End of the chat completion stream.");

    Ok(stream)
}

fn chat_stream_by_graph(
    graph: &mut Graph<GgmlMetadata>,
    id: impl Into<String>,
    include_usage: bool,
) -> Result<ChatStream, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Handle chat request with available tools by the model named {}.", graph.name());

    let id = id.into();

    match graph.compute() {
        Ok(_) => {
            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw generation:\n{}", output);

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                LlamaCoreError::Operation(format!("Failed to post-process the output. {}", e))
            })?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "post-processed generation:\n{}", &message);

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let usage = Some(Usage {
                prompt_tokens: token_info.prompt_tokens,
                completion_tokens: token_info.completion_tokens,
                total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
            });

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            if graph.metadata.prompt_template != PromptTemplateType::MistralTool
                && graph.metadata.prompt_template != PromptTemplateType::ChatMLTool
                && graph.metadata.prompt_template != PromptTemplateType::GroqLlama3Tool
                && graph.metadata.prompt_template != PromptTemplateType::Llama3Tool
                && graph.metadata.prompt_template != PromptTemplateType::InternLM2Tool
                && graph.metadata.prompt_template != PromptTemplateType::NemotronTool
                && graph.metadata.prompt_template != PromptTemplateType::FunctionaryV32
                && graph.metadata.prompt_template != PromptTemplateType::FunctionaryV31
                && graph.metadata.prompt_template != PromptTemplateType::MistralSmallTool
            {
                let err_msg = format!("Unsupported prompt template: {}. The tool use is only supported for 'mistral-tool', 'chatml-tool', 'groq-llama3-tool', 'llama-3-tool', 'internlm-2-tool', 'nemotron-tool', 'functionary-31', 'functionary-32', and 'mistral-small-tool' prompt templates.", graph.metadata.prompt_template);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                return Err(LlamaCoreError::Operation(err_msg));
            }

            let parsed_result = parse_tool_calls(&message, graph.metadata.prompt_template)?;

            let content = match parsed_result.content {
                Some(content) => Some(content),
                None => Some(parsed_result.raw),
            };

            let tool_calls: Vec<ToolCallForChunk> = parsed_result
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, tool_call)| ToolCallForChunk {
                    index,
                    id: tool_call.id,
                    ty: tool_call.ty,
                    function: tool_call.function,
                })
                .collect();

            // tool_calls chunk
            let tool_call_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkChoiceDelta {
                            role: ChatCompletionRole::Assistant,
                            content,
                            tool_calls,
                        },
                        logprobs: None,
                        finish_reason: None,
                    }],
                    usage: None,
                };
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // uage chunk
            let usage_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![],
                    usage,
                };
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // ending chunk
            let ending_chunk = "data: [DONE]\n\n".to_string();

            let chunks = vec![tool_call_chunk, usage_chunk, ending_chunk];

            Ok(ChatStream::new(
                Some(graph.name().to_owned()),
                id,
                include_usage,
                Some(chunks),
            ))
        }
        Err(wasmedge_wasi_nn::Error::BackendError(wasmedge_wasi_nn::BackendError::ContextFull)) => {
            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                let err_msg = format!("Failed to post-process the output. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let usage = Some(Usage {
                prompt_tokens: token_info.prompt_tokens,
                completion_tokens: token_info.completion_tokens,
                total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
            });

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            // context full chunk
            let context_full_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkChoiceDelta {
                            role: ChatCompletionRole::Assistant,
                            content: Some(message),
                            tool_calls: vec![],
                        },
                        logprobs: None,
                        finish_reason: Some(FinishReason::length),
                    }],
                    usage: None,
                };

                // serialize chat completion chunk
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // usage chunk
            let usage_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![],
                    usage,
                };

                // serialize chat completion chunk
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // ending chunk
            let ending_chunk = "data: [DONE]\n\n".to_string();

            let chunks = vec![context_full_chunk, usage_chunk, ending_chunk];

            Ok(ChatStream::new(
                Some(graph.name().to_owned()),
                id,
                include_usage,
                Some(chunks),
            ))
        }
        Err(wasmedge_wasi_nn::Error::BackendError(
            wasmedge_wasi_nn::BackendError::PromptTooLong,
        )) => {
            #[cfg(feature = "logging")]
            warn!(target: "stdout", "The prompt is too long. Please reduce the length of your input and try again.");

            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                let err_msg = format!("Failed to post-process the output. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // retrieve the number of prompt and completion token
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let usage = Some(Usage {
                prompt_tokens: token_info.prompt_tokens,
                completion_tokens: token_info.completion_tokens,
                total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
            });

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            // prompt too long chunk
            let prompt_too_long_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkChoiceDelta {
                            role: ChatCompletionRole::Assistant,
                            content: Some(message),
                            tool_calls: vec![],
                        },
                        logprobs: None,
                        finish_reason: Some(FinishReason::length),
                    }],
                    usage: None,
                };

                // serialize chat completion chunk
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // usage chunk
            let usage_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![],
                    usage,
                };

                // serialize chat completion chunk
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // ending chunk
            let ending_chunk = "data: [DONE]\n\n".to_string();

            let chunks = vec![prompt_too_long_chunk, usage_chunk, ending_chunk];

            Ok(ChatStream::new(
                Some(graph.name().to_owned()),
                id,
                include_usage,
                Some(chunks),
            ))
        }
        Err(e) => {
            let err_msg = format!("Failed to compute the chat completion. Reason: {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            Err(LlamaCoreError::Backend(BackendError::Compute(err_msg)))
        }
    }
}

async fn chat_once(
    chat_request: &mut ChatCompletionRequest,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Processing chat completion request in non-stream mode");

    let running_mode = running_mode()?;
    if !running_mode.contains(RunningMode::CHAT) && !running_mode.contains(RunningMode::RAG) {
        let err_msg = "The chat completion is only supported in the chat or rag mode.";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::Operation(err_msg.to_string()));
    }

    let model_name = chat_request.model.clone();
    let id = match &chat_request.user {
        Some(id) => id.clone(),
        None => gen_chat_id(),
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "user: {}", &id);

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Check model metadata");

    // update metadata
    let mut metadata = check_model_metadata(chat_request).await?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Build the chat prompt");

    // build prompt
    let (prompt, avaible_completion_tokens, tool_use) =
        build_prompt(model_name.as_ref(), chat_request)?;

    #[cfg(feature = "logging")]
    {
        info!(target: "stdout", "prompt:\n{}", &prompt);
        info!(target: "stdout", "available_completion_tokens: {}", avaible_completion_tokens);
        info!(target: "stdout", "tool_use: {}", tool_use);
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Update n_predict");

    // update metadata n_predict
    update_n_predict(chat_request, &mut metadata, avaible_completion_tokens).await?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Feed the prompt to the model");

    // feed the prompt to the model
    set_prompt(model_name.as_ref(), &prompt)?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute chat completion.");

    // compute
    let res = compute(model_name.as_ref(), id, tool_use);

    #[cfg(feature = "logging")]
    info!(target: "stdout", "End of the chat completion");

    res
}

fn compute(
    model_name: Option<&String>,
    id: impl Into<String>,
    tool_use: bool,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let mut chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match model_name {
        Some(model_name) => match chat_graphs.contains_key(model_name) {
            true => {
                let graph = chat_graphs.get_mut(model_name).unwrap();
                compute_by_graph(graph, id, tool_use)
            }
            false => match chat_graphs.iter_mut().next() {
                Some((_, graph)) => compute_by_graph(graph, id, tool_use),
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            },
        },
        None => match chat_graphs.iter_mut().next() {
            Some((_, graph)) => compute_by_graph(graph, id, tool_use),
            None => {
                let err_msg = "There is no model available in the chat graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                Err(LlamaCoreError::Operation(err_msg.into()))
            }
        },
    }
}

fn compute_by_graph(
    graph: &mut Graph<GgmlMetadata>,
    id: impl Into<String>,
    tool_use: bool,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute chat completion by the model named {}.", graph.name());

    match graph.compute() {
        Ok(_) => {
            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw generation: {}", output);

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                LlamaCoreError::Operation(format!("Failed to post-process the output. {}", e))
            })?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "post-processed generation:\n{}", &message);

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            match tool_use {
                true => {
                    if graph.metadata.prompt_template != PromptTemplateType::MistralTool
                        && graph.metadata.prompt_template != PromptTemplateType::ChatMLTool
                        && graph.metadata.prompt_template != PromptTemplateType::GroqLlama3Tool
                        && graph.metadata.prompt_template != PromptTemplateType::Llama3Tool
                        && graph.metadata.prompt_template != PromptTemplateType::InternLM2Tool
                        && graph.metadata.prompt_template != PromptTemplateType::NemotronTool
                        && graph.metadata.prompt_template != PromptTemplateType::FunctionaryV32
                        && graph.metadata.prompt_template != PromptTemplateType::FunctionaryV31
                        && graph.metadata.prompt_template != PromptTemplateType::MistralSmallTool
                    {
                        let err_msg = format!("Unsupported prompt template: {}. The tool use is only supported for 'mistral-tool', 'chatml-tool', 'groq-llama3-tool', 'llama-3-tool', 'internlm-2-tool', 'nemotron-tool', 'functionary-31', 'functionary-32', and 'mistral-small-tool' prompt templates.", graph.metadata.prompt_template);

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        return Err(LlamaCoreError::Operation(err_msg));
                    }

                    let parsed_result = parse_tool_calls(&message, graph.metadata.prompt_template)?;

                    let finish_reason = if parsed_result.tool_calls.is_empty() {
                        FinishReason::stop
                    } else {
                        FinishReason::tool_calls
                    };

                    let content = match parsed_result.content {
                        Some(content) => Some(content),
                        None => Some(parsed_result.raw),
                    };

                    // create ChatCompletionResponse
                    Ok(ChatCompletionObject {
                        id: id.into(),
                        object: String::from("chat.completion"),
                        created: created.as_secs(),
                        model: graph.name().to_owned(),
                        choices: vec![ChatCompletionObjectChoice {
                            index: 0,
                            message: ChatCompletionObjectMessage {
                                role: ChatCompletionRole::Assistant,
                                content,
                                tool_calls: parsed_result.tool_calls,
                                function_call: None,
                            },
                            finish_reason,
                            logprobs: None,
                        }],
                        usage: Usage {
                            prompt_tokens: token_info.prompt_tokens,
                            completion_tokens: token_info.completion_tokens,
                            total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
                        },
                    })
                }
                false => {
                    // create ChatCompletionResponse
                    Ok(ChatCompletionObject {
                        id: id.into(),
                        object: String::from("chat.completion"),
                        created: created.as_secs(),
                        model: graph.name().to_owned(),
                        choices: vec![ChatCompletionObjectChoice {
                            index: 0,
                            message: ChatCompletionObjectMessage {
                                role: ChatCompletionRole::Assistant,
                                content: Some(message),
                                tool_calls: vec![],
                                function_call: None,
                            },
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
            }
        }
        Err(wasmedge_wasi_nn::Error::BackendError(wasmedge_wasi_nn::BackendError::ContextFull)) => {
            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                let err_msg = format!("Failed to post-process the output. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            // create ChatCompletionResponse
            Ok(ChatCompletionObject {
                id: id.into(),
                object: String::from("chat.completion"),
                created: created.as_secs(),
                model: graph.name().to_owned(),
                choices: vec![ChatCompletionObjectChoice {
                    index: 0,
                    message: ChatCompletionObjectMessage {
                        role: ChatCompletionRole::Assistant,
                        content: Some(message),
                        tool_calls: vec![],
                        function_call: None,
                    },
                    finish_reason: FinishReason::length,
                    logprobs: None,
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
            #[cfg(feature = "logging")]
            warn!(target: "stdout", "The prompt is too long. Please reduce the length of your input and try again.");

            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                let err_msg = format!("Failed to post-process the output. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // retrieve the number of prompt and completion token
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            // create ChatCompletionResponse
            Ok(ChatCompletionObject {
                id: id.into(),
                object: String::from("chat.completion"),
                created: created.as_secs(),
                model: graph.name().to_owned(),
                choices: vec![ChatCompletionObjectChoice {
                    index: 0,
                    message: ChatCompletionObjectMessage {
                        role: ChatCompletionRole::Assistant,
                        content: Some(message),
                        tool_calls: vec![],
                        function_call: None,
                    },
                    finish_reason: FinishReason::length,
                    logprobs: None,
                }],
                usage: Usage {
                    prompt_tokens: token_info.prompt_tokens,
                    completion_tokens: token_info.completion_tokens,
                    total_tokens: token_info.completion_tokens + token_info.completion_tokens,
                },
            })
        }
        Err(e) => {
            let err_msg = format!("Failed to compute the chat completion. Reason: {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            Err(LlamaCoreError::Backend(BackendError::Compute(err_msg)))
        }
    }
}

fn parse_tool_calls(
    input: &str,
    prompt_template: PromptTemplateType,
) -> Result<ParseResult, LlamaCoreError> {
    match prompt_template {
        PromptTemplateType::MistralTool => match regex::Regex::new(r"\[\{.*?\}\]") {
            Ok(re) => {
                let mut values: Vec<serde_json::Value> = vec![];
                for cap in re.captures_iter(input) {
                    let matched = &cap[0];

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "captured: {}", matched);

                    match serde_json::from_str::<Vec<serde_json::Value>>(matched) {
                        Ok(group) => values.extend(group),
                        Err(e) => {
                            let err_msg = format!(
                                "Failed to deserialize generated tool calls. Reason: {}",
                                e
                            );

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        }
                    }
                }

                let mut tool_calls: Vec<ToolCall> = vec![];
                for value in values.iter() {
                    let name = match value.get("name") {
                        Some(name) => name.to_string().replace("\"", ""),
                        None => {
                            let err_msg = format!(
                                "Failed to get the name of the function. Tool call: {:?}",
                                value
                            );

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        }
                    };

                    let arguments = match value.get("arguments") {
                        Some(arguments) => arguments.to_string(),
                        None => {
                            let err_msg = format!(
                                "Failed to get the arguments of the function. Tool call: {:?}",
                                value
                            );

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        }
                    };

                    let function = Function { name, arguments };

                    let tool_call = ToolCall {
                        id: "call_abc123".to_string(),
                        ty: "function".to_string(),
                        function,
                    };

                    tool_calls.push(tool_call);
                }

                let parsed = ParseResult {
                    raw: input.to_owned(),
                    content: None,
                    tool_calls,
                };

                #[cfg(feature = "logging")]
                info!(target: "stdout", "parsed result: {:?}", parsed);

                Ok(parsed)
            }
            Err(e) => {
                let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                Err(LlamaCoreError::Operation(err_msg))
            }
        },
        PromptTemplateType::ChatMLTool => {
            match regex::Regex::new(r"<tool_call>(.*?)</tool_call>") {
                Ok(re) => {
                    let mut values: Vec<serde_json::Value> = vec![];
                    for cap in re.captures_iter(input) {
                        let matched = cap[1].replace("\\n", ""); // Remove "\\n" from the captured group

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "captured: {}", &matched);

                        match serde_json::from_str::<serde_json::Value>(&matched) {
                            Ok(value) => values.push(value),
                            Err(e) => {
                                let err_msg = format!(
                                    "Failed to deserialize generated tool calls. Reason: {}",
                                    e
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    }

                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for value in values.iter() {
                        let name = match value.get("name") {
                            Some(name) => name.to_string().replace("\"", ""),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the name of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let arguments = match value.get("arguments") {
                            Some(arguments) => arguments.to_string(),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the arguments of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let function = Function { name, arguments };

                        let tool_call = ToolCall {
                            id: "call_abc123".to_string(),
                            ty: "function".to_string(),
                            function,
                        };

                        tool_calls.push(tool_call);
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        PromptTemplateType::GroqLlama3Tool => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            match regex::Regex::new(r"(?s)<tool_call>((.|\r|\n)*?)</tool_call>") {
                Ok(re) => {
                    let mut values: Vec<serde_json::Value> = vec![];
                    for cap in re.captures_iter(input) {
                        let matched = cap[1].trim();

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "captured: {}", matched);

                        match serde_json::from_str::<serde_json::Value>(matched) {
                            Ok(value) => values.push(value),
                            Err(e) => {
                                let err_msg = format!(
                                    "Failed to deserialize generated tool calls. Reason: {}",
                                    e
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    }

                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for value in values.iter() {
                        let name = match value.get("name") {
                            Some(name) => name.to_string().replace("\"", ""),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the name of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let arguments = match value.get("arguments") {
                            Some(arguments) => arguments.to_string(),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the arguments of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let function = Function { name, arguments };

                        let tool_call = ToolCall {
                            id: "call_abc123".to_string(),
                            ty: "function".to_string(),
                            function,
                        };

                        tool_calls.push(tool_call);
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        PromptTemplateType::Llama3Tool => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            let re = match regex::Regex::new(r"^\{(.|\r|\n)*\}$") {
                Ok(re) => re,
                Err(e) => {
                    let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg));
                }
            };

            if re.is_match(input) {
                match serde_json::from_str::<serde_json::Value>(input) {
                    Ok(value) => {
                        let values: Vec<serde_json::Value> = vec![value];

                        let mut tool_calls: Vec<ToolCall> = vec![];
                        for value in values.iter() {
                            let name = match value.get("name") {
                                Some(name) => name.to_string().replace("\"", ""),
                                None => {
                                    let err_msg = format!(
                                        "Failed to get the name of the function. Tool call: {:?}",
                                        value
                                    );

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", &err_msg);

                                    return Err(LlamaCoreError::Operation(err_msg));
                                }
                            };

                            let arguments = match value.get("parameters") {
                                Some(arguments) => arguments.to_string(),
                                None => {
                                    let err_msg = format!(
                                        "Failed to get the arguments of the function. Tool call: {:?}",
                                        value
                                    );

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", &err_msg);

                                    return Err(LlamaCoreError::Operation(err_msg));
                                }
                            };

                            let function = Function { name, arguments };

                            let tool_call = ToolCall {
                                id: "call_abc123".to_string(),
                                ty: "function".to_string(),
                                function,
                            };

                            tool_calls.push(tool_call);
                        }

                        let parsed = ParseResult {
                            raw: input.to_owned(),
                            content: None,
                            tool_calls,
                        };

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "parsed result: {:?}", parsed);

                        Ok(parsed)
                    }
                    Err(e) => {
                        let err_msg =
                            format!("Failed to deserialize generated tool calls. Reason: {}", e);

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        Err(LlamaCoreError::Operation(err_msg))
                    }
                }
            } else {
                let parsed = ParseResult {
                    raw: input.to_owned(),
                    content: None,
                    tool_calls: vec![],
                };

                #[cfg(feature = "logging")]
                info!(target: "stdout", "parsed result: {:?}", parsed);

                Ok(parsed)
            }
        }
        PromptTemplateType::InternLM2Tool => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            let blocks: Vec<&str> = input.trim().split("<|action_start|><|plugin|>").collect();

            #[cfg(feature = "logging")]
            info!(target: "stdout", "blocks: {:?}", blocks);

            let mut tool_calls: Vec<ToolCall> = vec![];
            let mut content = String::new();
            for block in blocks {
                let block = block.trim();
                if !block.is_empty() {
                    if block.ends_with("<|action_end|>") {
                        let value = block.trim().trim_end_matches("<|action_end|>");

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "tool call: {}", value);

                        match serde_json::from_str::<serde_json::Value>(value) {
                            Ok(value) => {
                                let name = match value.get("name") {
                                    Some(name) => name.to_string().replace("\"", ""),
                                    None => {
                                        let err_msg = format!(
                                            "Failed to get the name of the function. Tool call: {:?}",
                                            value
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        return Err(LlamaCoreError::Operation(err_msg));
                                    }
                                };

                                let arguments = match value.get("parameters") {
                                    Some(arguments) => arguments.to_string(),
                                    None => {
                                        let err_msg = format!(
                                            "Failed to get the arguments of the function. Tool call: {:?}",
                                            value
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        return Err(LlamaCoreError::Operation(err_msg));
                                    }
                                };

                                let function = Function { name, arguments };

                                let tool_call = ToolCall {
                                    id: "call_abc123".to_string(),
                                    ty: "function".to_string(),
                                    function,
                                };

                                tool_calls.push(tool_call);
                            }
                            Err(e) => {
                                let err_msg = format!(
                                    "Failed to deserialize generated tool calls. Reason: {}",
                                    e
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    } else {
                        content.push_str(block);
                        content.push('\n');
                    }
                }
            }

            let parsed = match content.is_empty() {
                true => ParseResult {
                    raw: input.to_owned(),
                    content: None,
                    tool_calls,
                },
                false => ParseResult {
                    raw: input.to_owned(),
                    content: Some(content.trim().to_owned()),
                    tool_calls,
                },
            };

            #[cfg(feature = "logging")]
            info!(target: "stdout", "parsed result: {:?}", parsed);

            Ok(parsed)
        }
        PromptTemplateType::NemotronTool => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            match regex::Regex::new(r"(?s)<toolcall>\s*(.*?)\s*</toolcall>") {
                Ok(re) => {
                    let mut values: Vec<serde_json::Value> = vec![];
                    for cap in re.captures_iter(input) {
                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "captured: {}", &cap[0]);

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "extracted: {}", &cap[1]);

                        let matched = cap[1].trim();

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "captured: {}", matched);

                        match serde_json::from_str::<serde_json::Value>(matched) {
                            Ok(value) => values.push(value),
                            Err(e) => {
                                let err_msg = format!(
                                    "Failed to deserialize generated tool calls. Reason: {}",
                                    e
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    }

                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for value in values.iter() {
                        let name = match value.get("name") {
                            Some(name) => name.to_string().replace("\"", ""),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the name of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let arguments = match value.get("arguments") {
                            Some(arguments) => arguments.to_string(),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the arguments of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let function = Function { name, arguments };

                        let tool_call = ToolCall {
                            id: "call_abc123".to_string(),
                            ty: "function".to_string(),
                            function,
                        };

                        tool_calls.push(tool_call);
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        PromptTemplateType::FunctionaryV32 => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            match regex::Regex::new(r">>>\s*(\w+)\s*\{(.*)\}<\|eot_id\|>") {
                Ok(re) => {
                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for cap in re.captures_iter(input) {
                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "func_name: {}", &cap[1]);

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "arguments: {}", &cap[2]);

                        let tool_call = ToolCall {
                            id: "call_abc123".to_string(),
                            ty: "function".to_string(),
                            function: Function {
                                name: cap[1].to_string(),
                                arguments: cap[2].to_string(),
                            },
                        };

                        tool_calls.push(tool_call);
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let warn_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    warn!(target: "stdout", "{}", &warn_msg);

                    Ok(ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls: vec![],
                    })
                }
            }
        }
        PromptTemplateType::FunctionaryV31 => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            match regex::Regex::new(r"<function=(\w+)>\s*(\{.*?\})</function>") {
                Ok(re) => {
                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for cap in re.captures_iter(input) {
                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "func_name: {}", &cap[1]);

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "arguments: {}", &cap[2]);

                        let tool_call = ToolCall {
                            id: "call_abc123".to_string(),
                            ty: "function".to_string(),
                            function: Function {
                                name: cap[1].to_string(),
                                arguments: cap[2].to_string(),
                            },
                        };

                        tool_calls.push(tool_call);
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let warn_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    warn!(target: "stdout", "{}", &warn_msg);

                    Ok(ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls: vec![],
                    })
                }
            }
        }
        PromptTemplateType::MistralSmallTool => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            match regex::Regex::new(r"\[TOOL_CALLS\]\s*(\[(.*?)\])") {
                Ok(re) => {
                    let mut values: Vec<serde_json::Value> = vec![];
                    if let Some(cap) = re.captures(input) {
                        let matched = cap[1].trim();

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "captured: {}", matched);

                        match serde_json::from_str::<Vec<serde_json::Value>>(matched) {
                            Ok(vals) => values = vals,
                            Err(e) => {
                                let err_msg = format!(
                                    "Failed to deserialize generated tool calls. Reason: {}",
                                    e
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    };

                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for value in values.iter() {
                        if let Some(object_map) = value.as_object() {
                            if object_map.contains_key("function") {
                                let mut function = Function {
                                    name: String::new(),
                                    arguments: String::new(),
                                };

                                let value = object_map.get("function").unwrap();
                                let func_map = value.as_object().unwrap();
                                if func_map.contains_key("name") {
                                    let func_name = func_map.get("name").unwrap().as_str().unwrap();
                                    println!("Function name: {:?}", func_name);

                                    function.name = func_name.to_string();
                                }
                                if func_map.contains_key("arguments") {
                                    let args = func_map.get("arguments").unwrap();
                                    let arguments = args.to_string();
                                    println!("Arguments: {:?}", arguments);

                                    function.arguments = arguments;
                                }

                                let tool_call = ToolCall {
                                    id: "call_abc123".to_string(),
                                    ty: "function".to_string(),
                                    function,
                                };

                                tool_calls.push(tool_call);
                            } else if object_map.contains_key("name") {
                                let mut function = Function {
                                    name: String::new(),
                                    arguments: String::new(),
                                };

                                let name = object_map.get("name").unwrap().as_str().unwrap();
                                println!("name: {:?}", name);
                                function.name = name.to_string();

                                if object_map.contains_key("arguments") {
                                    let args = object_map.get("arguments").unwrap();
                                    let arguments = args.to_string();
                                    println!("Arguments: {:?}", arguments);

                                    function.arguments = arguments;
                                }

                                let tool_call = ToolCall {
                                    id: "call_abc123".to_string(),
                                    ty: "function".to_string(),
                                    function,
                                };

                                tool_calls.push(tool_call);
                            }
                        }

                        // match value.get("function") {
                        //     Some(func) => {
                        //         let name = match func.get("name") {
                        //             Some(name) => name.as_str().unwrap().to_string(),
                        //             None => String::new(),
                        //         };

                        //         let arguments = match func.get("arguments") {
                        //             Some(arguments) => arguments.to_string(),
                        //             None => String::new(),
                        //         };

                        //         let function = Function { name, arguments };

                        //         let tool_call = ToolCall {
                        //             id: "call_abc123".to_string(),
                        //             ty: "function".to_string(),
                        //             function,
                        //         };

                        //         tool_calls.push(tool_call);
                        //     }
                        //     None => {
                        //         let name = match value.get("name") {
                        //             Some(name) => name.as_str().unwrap().to_string(),
                        //             None => String::new(),
                        //         };

                        //         let arguments = match value.get("arguments") {
                        //             Some(arguments) => arguments.to_string(),
                        //             None => String::new(),
                        //         };

                        //         let function = Function { name, arguments };

                        //         let tool_call = ToolCall {
                        //             id: "call_abc123".to_string(),
                        //             ty: "function".to_string(),
                        //             function,
                        //         };

                        //         tool_calls.push(tool_call);
                        //     }
                        // }
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        _ => {
            let err_msg = format!(
                "The tool use is only supported for prompt templates: {}, {}, {}, {}, {}, {}, {}, and {}.",
                PromptTemplateType::MistralTool,
                PromptTemplateType::ChatMLTool,
                PromptTemplateType::GroqLlama3Tool,
                PromptTemplateType::Llama3Tool,
                PromptTemplateType::InternLM2Tool,
                PromptTemplateType::NemotronTool,
                PromptTemplateType::FunctionaryV32,
                PromptTemplateType::MistralSmallTool,
            );

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            Err(LlamaCoreError::Operation(err_msg))
        }
    }
}

async fn check_model_metadata(
    chat_request: &ChatCompletionRequest,
) -> Result<GgmlMetadata, LlamaCoreError> {
    let mut should_update = false;
    let mut metadata = get_model_metadata(chat_request.model.as_ref())?;

    // check if necessary to update `image`
    if metadata.prompt_template.is_image_supported() {
        if let Some(ChatCompletionRequestMessage::User(user_message)) = chat_request.messages.last()
        {
            if let ChatCompletionUserMessageContent::Parts(parts) = user_message.content() {
                for part in parts {
                    if let ContentPart::Image(image_part) = part {
                        let image = image_part.image();

                        if image.is_url() {
                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "The image is provided in URL format.");

                            // download the image
                            let img_path_str = download_image(&image.url).await?;

                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "The image is saved to {}", img_path_str);

                            // update metadata image
                            metadata.image = Some(img_path_str);

                            if !should_update {
                                should_update = true;
                            }

                            // TODO: now only support a single image

                            break;
                        } else {
                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "The image is provided in base64 format.");

                            // TODO: now only support a single image

                            break;
                        }
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
        #[cfg(feature = "logging")]
        info!(target: "stdout", "Update the model metadata.");

        // update the target graph with the new metadata
        update_model_metadata(chat_request.model.as_ref(), &metadata)?;
    }

    Ok(metadata)
}

async fn update_n_predict(
    chat_request: &ChatCompletionRequest,
    metadata: &mut GgmlMetadata,
    available_completion_tokens: u64,
) -> Result<(), LlamaCoreError> {
    let mut should_update = false;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "n_predict: {}", metadata.n_predict);

    // From high to low priority
    // 1. chat_request.max_tokens & chat_request.max_completion_tokens
    // 2. available_completion_tokens
    // 3. n_predict

    if let Some(max_completion_tokens) = chat_request.max_completion_tokens {
        if metadata.n_predict != max_completion_tokens {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Update n_predict with max_completion_tokens from {} to {}", metadata.n_predict, max_completion_tokens);

            metadata.n_predict = max_completion_tokens;

            if !should_update {
                should_update = true;
            }
        }
    }

    // TODO: remove this condition after [Issue #3958 on WasmEdge](https://github.com/WasmEdge/WasmEdge/issues/3958) is fixed
    if metadata.n_predict == -2 {
        #[cfg(feature = "logging")]
        info!(target: "stdout", "Update n_predict with available_completion_tokens from {} to {}", metadata.n_predict, available_completion_tokens);

        // update n_predict
        metadata.n_predict = available_completion_tokens as i32;

        if !should_update {
            should_update = true;
        }
    }

    if metadata.n_predict == -1
        || (metadata.n_predict > 0 && metadata.n_predict < available_completion_tokens as i32)
        || (metadata.n_predict < 0 && metadata.n_predict != -2)
    // TODO: remove this condition after [Issue #3958 on WasmEdge](https://github.com/WasmEdge/WasmEdge/issues/3958) is fixed
    {
        #[cfg(feature = "logging")]
        info!(target: "stdout", "Update n_predict with available_completion_tokens from {} to {}", metadata.n_predict, available_completion_tokens);

        // update n_predict
        metadata.n_predict = available_completion_tokens as i32;

        if !should_update {
            should_update = true;
        }
    }

    if should_update {
        #[cfg(feature = "logging")]
        info!(target: "stdout", "Update the model metadata.");

        // update the target graph with the new metadata
        update_model_metadata(chat_request.model.as_ref(), metadata)?;
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
    } else if *template_ty == PromptTemplateType::GemmaInstruct {
        let s = output.as_ref().trim();
        if s.ends_with("<end_of_turn>") {
            s.trim_end_matches("<end_of_turn>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::ChatML
        || *template_ty == PromptTemplateType::ChatMLTool
        || *template_ty == PromptTemplateType::InternLM2Tool
        || *template_ty == PromptTemplateType::MiniCPMV
    {
        let mut s = output.as_ref().trim();
        if s.ends_with("<|endoftext|>") {
            s = s.trim_end_matches("<|endoftext|>").trim();
        }

        if s.contains("<|im_start|>") && s.contains("<|im_end|>") {
            let idx_start = s.find("<|im_start|>").unwrap();
            let idx_end = s.find("<|im_end|>").unwrap();

            match idx_start <= idx_end {
                true => s.split("<|im_start|>").collect::<Vec<_>>()[0]
                    .trim()
                    .to_owned(),
                false => s.split("<|im_end|>").collect::<Vec<_>>()[0]
                    .trim()
                    .to_owned(),
            }
        } else if s.contains("<|im_start|>") {
            s.split("<|im_start|>").collect::<Vec<_>>()[0]
                .trim()
                .to_owned()
        } else if s.contains("<|im_end|>") {
            let output = s.trim_end_matches("<|im_end|>").trim();
            if output.starts_with(": ") {
                output.trim_start_matches(": ").to_owned()
            } else {
                output.to_owned()
            }
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Zephyr
        || *template_ty == PromptTemplateType::MistralLite
        || *template_ty == PromptTemplateType::MistralTool
        || *template_ty == PromptTemplateType::MistralInstruct
        || *template_ty == PromptTemplateType::MistralSmallChat
        || *template_ty == PromptTemplateType::MistralSmallTool
        || *template_ty == PromptTemplateType::BreezeInstruct
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
    } else if *template_ty == PromptTemplateType::Llama2Chat
        || *template_ty == PromptTemplateType::NemotronTool
        || *template_ty == PromptTemplateType::NemotronChat
    {
        let s = output.as_ref().trim();
        if s.ends_with("</s>") {
            s.trim_end_matches("</s>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Llama3Chat
        || *template_ty == PromptTemplateType::GroqLlama3Tool
        || *template_ty == PromptTemplateType::Llama3Tool
        || *template_ty == PromptTemplateType::FunctionaryV32
    {
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
    } else if *template_ty == PromptTemplateType::Phi4Chat {
        let s = output.as_ref().trim();
        if s.ends_with("<|im_end|>") {
            s.trim_end_matches("<|im_end|>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::FunctionaryV31 {
        let mut s = output.as_ref().trim();
        if s.ends_with("<|eot_id|>") {
            s = s.trim_end_matches("<|eot_id|>").trim();
        }
        if s.ends_with("<|eom_id|>") {
            s = s.trim_end_matches("<|eom_id|>").trim();
        }
        s.to_owned()
    } else if *template_ty == PromptTemplateType::MoxinChat {
        let s = output.as_ref().trim();
        if s.ends_with("</s>") {
            s.trim_end_matches("</s>").trim().to_owned()
        } else if s.ends_with("[INST]") {
            s.trim_end_matches("[INST]").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Falcon3 {
        let s = output.as_ref().trim();
        if s.ends_with("<|endoftext|>") {
            s.trim_end_matches("<|endoftext|>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Megrez {
        let s = output.as_ref().trim();
        if s.ends_with("<|turn_end|>") {
            s.trim_end_matches("<|turn_end|>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Qwen2vl {
        let mut s = output.as_ref().trim();

        if s.starts_with(":") {
            s = s.trim_start_matches(":").trim();
        }

        if s.ends_with("<|im_end|>") {
            s.trim_end_matches("<|im_end|>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::VicunaLlava {
        let s = output.as_ref().trim();
        if s.ends_with("</s>") {
            s.trim_end_matches("</s>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else {
        output.as_ref().trim().to_owned()
    };

    Ok(output)
}

/// Build the chat prompt from the chat messages.
///
/// # Arguments
///
/// * `model_name`: The name of the model.
///
/// * `chat_request`: The chat request.
///
/// # Returns
///
/// A tuple containing the prompt, the number of available tokens for completions, and a boolean indicating whether tools are used.
fn build_prompt(
    model_name: Option<&String>,
    chat_request: &mut ChatCompletionRequest,
) -> Result<(String, u64, bool), LlamaCoreError> {
    let metadata = get_model_metadata(model_name)?;
    let ctx_size = metadata.ctx_size as u64;
    let chat_prompt = ChatPrompt::from(metadata.prompt_template);

    // compute max prompt tokens, which is 80% of the context size
    let max_prompt_tokens = ctx_size * 4 / 5;

    loop {
        // ! DO NOT REMOVE
        {
            // // build prompt
            // let prompt = match chat_prompt.build(&mut chat_request.messages) {
            //     Ok(prompt) => prompt,
            //     Err(e) => {
            //         let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

            //         #[cfg(feature = "logging")]
            //         error!(target: "stdout", "{}", &err_msg);

            //         return Err(LlamaCoreError::Operation(err_msg));
            //     }
            // };
        }

        if chat_request.messages.is_empty() {
            let err_msg = "The messages in the chat request are empty.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.to_owned()));
        }

        #[cfg(feature = "logging")]
        {
            let mut role_chain = String::new();
            for (idx, message) in chat_request.messages.iter().enumerate() {
                if idx == chat_request.messages.len() - 1 {
                    role_chain.push_str(&format!("{}", message.role()));
                } else {
                    role_chain.push_str(&format!("{} -> ", message.role()));
                }
            }
            info!(target: "stdout", "Role chain: {}", role_chain);
        }

        let (prompt, tool_use) = match chat_request.tool_choice.as_ref() {
            Some(tool_choice) => match tool_choice {
                ToolChoice::None => {
                    match chat_prompt.build_with_tools(&mut chat_request.messages, Some(&[])) {
                        Ok(prompt) => (prompt, false),
                        Err(e) => {
                            let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        }
                    }
                }
                _ => match chat_request.tools.as_ref() {
                    Some(tools) => match chat_prompt
                        .build_with_tools(&mut chat_request.messages, Some(tools.as_slice()))
                    {
                        Ok(prompt) => (prompt, true),
                        Err(e) => {
                            let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        }
                    },
                    None => {
                        #[cfg(feature = "logging")]
                        warn!(target: "stdout", "The tool choice without tools is not supported.");

                        match chat_prompt.build_with_tools(&mut chat_request.messages, None) {
                            Ok(prompt) => (prompt, false),
                            Err(e) => {
                                let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    }
                },
            },
            None => match chat_prompt.build_with_tools(&mut chat_request.messages, None) {
                Ok(prompt) => (prompt, false),
                Err(e) => {
                    let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg));
                }
            },
        };
        #[cfg(feature = "logging")]
        info!(target: "stdout", "Try to set prompt: {}", prompt);

        // set prompt
        set_prompt(model_name, &prompt)?;

        // Retrieve the number of prompt tokens.
        let token_info = get_token_info_by_graph_name(model_name)?;

        match token_info.prompt_tokens > max_prompt_tokens {
            true => {
                match chat_request.messages[0].role() {
                    ChatCompletionRole::System => {
                        if chat_request.messages.len() > 2 {
                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "Prune chat history: current length {}", chat_request.messages.len());

                            // remove user_1 if it exists
                            // For example, `system -> user_1 -> ... -> user_2 -> ... -> user_latest` will be converted to `system -> ... -> user_2 -> ... -> user_latest`
                            if chat_request.messages[1].role() == ChatCompletionRole::User {
                                let user_message = chat_request.messages.remove(1);

                                #[cfg(feature = "logging")]
                                info!(target: "stdout", "Remove a user message from the chat history: {:?}", user_message);
                            }

                            // remove all messages until the message is of `user`
                            // For example, `system -> ... -> user_2 -> ... -> user_latest` will be converted to `system -> user_2 -> ... -> user_latest`
                            while chat_request.messages[1].role() != ChatCompletionRole::User {
                                let message = chat_request.messages.remove(1);

                                #[cfg(feature = "logging")]
                                info!(target: "stdout", "Remove a {} message from the chat history: {:?}", message.role(), message);

                                if chat_request.messages.len() == 1 {
                                    let err_msg = format!("The last message in the chat history should be a user message, but found a {} message.", message.role());

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", err_msg);

                                    return Err(LlamaCoreError::Operation(err_msg));
                                }
                            }
                        } else if token_info.prompt_tokens > ctx_size {
                            let err_msg = format!(
                                    "The number of prompt tokens ({}) is greater than the context size ({}). Please increase the context size, or simplify the input message.",
                                    token_info.prompt_tokens, ctx_size
                                );

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        } else {
                            return Ok((prompt, ctx_size - token_info.prompt_tokens, tool_use));
                        }
                    }
                    ChatCompletionRole::User => {
                        if chat_request.messages.len() > 1 {
                            // user_1 -> ... -> user_2 -> ... -> user_latest

                            // remove user_1 if it exists
                            // For example, `user_1 -> ... -> user_2 -> ... -> user_latest` will be converted to `... -> user_2 -> ... -> user_latest`
                            if chat_request.messages[0].role() == ChatCompletionRole::User {
                                let user_message = chat_request.messages.remove(0);

                                #[cfg(feature = "logging")]
                                info!(target: "stdout", "Remove a user message from the chat history: {:?}", user_message);
                            }

                            // remove all messages until the message is of `user`
                            // For example, `... -> user_2 -> ... -> user_latest` will be converted to `user_2 -> ... -> user_latest`
                            while chat_request.messages[0].role() != ChatCompletionRole::User {
                                let message = chat_request.messages.remove(0);

                                #[cfg(feature = "logging")]
                                info!(target: "stdout", "Remove a {} message from the chat history: {:?}", message.role(), message);

                                if chat_request.messages.is_empty() {
                                    let err_msg = format!("The last message in the chat history should be a user message, but found a {} message.", message.role());

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", err_msg);

                                    return Err(LlamaCoreError::Operation(err_msg));
                                }
                            }
                        } else if token_info.prompt_tokens > ctx_size {
                            let err_msg = format!(
                                    "The number of prompt tokens ({}) is greater than the context size ({}). Please increase the context size, or simplify the input message.",
                                    token_info.prompt_tokens, ctx_size
                                );

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        } else {
                            return Ok((prompt, ctx_size - token_info.prompt_tokens, tool_use));
                        }
                    }
                    _ => {
                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "remove a {} message from the message queue", chat_request.messages[0].role());

                        chat_request.messages.remove(0);
                    }
                }

                continue;
            }
            false => return Ok((prompt, ctx_size - max_prompt_tokens, tool_use)),
        }
    }
}

/// Downloads an image from the given URL and returns the file name.
async fn download_image(image_url: impl AsRef<str>) -> Result<String, LlamaCoreError> {
    let image_url = image_url.as_ref();
    let url = reqwest::Url::parse(image_url).map_err(|e| {
        let err_msg = format!("Fail to parse the image URL: {}. Reason: {}", image_url, e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let response = reqwest::get(url).await.map_err(|e| {
        let err_msg = format!(
            "Fail to download the image from the URL: {}. Reason: {}",
            image_url, e
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let fname = response
        .url()
        .path_segments()
        .and_then(|mut segments| segments.next_back())
        .and_then(|name| if name.is_empty() { None } else { Some(name) })
        .ok_or(LlamaCoreError::Operation(format!(
            "Fail to get the file name: {}",
            image_url
        )))?
        .to_string();

    // create a unique file id
    let id = format!("file_{}", uuid::Uuid::new_v4());
    // save the file
    let path = Path::new("archives");
    if !path.exists() {
        fs::create_dir(path).unwrap();
    }
    let file_path = path.join(&id);
    if !file_path.exists() {
        fs::create_dir(&file_path).unwrap();
    }
    let img_path = file_path.join(&fname);
    let mut dest: File = File::create(&img_path).map_err(|e| {
        let err_msg = format!("Failed to create archive document {}. {}", &fname, e);

        // log
        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let mut content = response.bytes_stream();
    while let Some(Ok(item)) = content.next().await {
        std::io::copy(&mut item.as_ref(), &mut dest).map_err(|e| {
            let err_msg = format!(
                "Fail to write the image content to the file: {}. Reason: {}",
                &fname, e
            );

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;
    }

    Ok(img_path.as_path().to_string_lossy().to_string())
}

fn set_prompt(model_name: Option<&String>, prompt: impl AsRef<str>) -> Result<(), LlamaCoreError> {
    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let mut chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match model_name {
        Some(model_name) => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Set prompt to the chat model named {}", model_name);

            match chat_graphs.contains_key(model_name) {
                true => {
                    let graph = chat_graphs.get_mut(model_name).unwrap();
                    let tensor_data = prompt.as_ref().as_bytes().to_vec();
                    set_tensor_data_u8(graph, 0, &tensor_data)
                }
                false => match chat_graphs.iter_mut().next() {
                    Some((_, graph)) => {
                        let tensor_data = prompt.as_ref().as_bytes().to_vec();
                        set_tensor_data_u8(graph, 0, &tensor_data)
                    }
                    None => {
                        let err_msg = "There is no model available in the chat graphs.";

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        Err(LlamaCoreError::Operation(err_msg.into()))
                    }
                },
            }
        }
        None => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Set prompt to the default chat model.");

            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    let tensor_data = prompt.as_ref().as_bytes().to_vec();
                    set_tensor_data_u8(graph, 0, &tensor_data)
                }
                None => {
                    let err_msg = "There is no model available in the chat graphs while trying to set prompt to the default model.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    }
}

/// Get a copy of the metadata of the model.
fn get_model_metadata(model_name: Option<&String>) -> Result<GgmlMetadata, LlamaCoreError> {
    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match model_name {
        Some(model_name) => match chat_graphs.contains_key(model_name) {
            true => {
                let graph = chat_graphs.get(model_name).unwrap();
                Ok(graph.metadata.clone())
            }
            false => match chat_graphs.iter().next() {
                Some((_, graph)) => Ok(graph.metadata.clone()),
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            },
        },
        None => match chat_graphs.iter().next() {
            Some((_, graph)) => Ok(graph.metadata.clone()),
            None => {
                let err_msg = "There is no model available in the chat graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                Err(LlamaCoreError::Operation(err_msg.into()))
            }
        },
    }
}

fn update_model_metadata(
    model_name: Option<&String>,
    metadata: &GgmlMetadata,
) -> Result<(), LlamaCoreError> {
    let config = match serde_json::to_string(metadata) {
        Ok(config) => config,
        Err(e) => {
            let err_msg = format!("Fail to serialize metadata to a JSON string. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg));
        }
    };

    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let mut chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. Reason: {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match model_name {
        Some(model_name) => {
            match chat_graphs.contains_key(model_name) {
                true => {
                    let graph = chat_graphs.get_mut(model_name).unwrap();
                    // update metadata
                    set_tensor_data_u8(graph, 1, config.as_bytes())
                }
                false => match chat_graphs.iter_mut().next() {
                    Some((_, graph)) => {
                        // update metadata
                        set_tensor_data_u8(graph, 1, config.as_bytes())
                    }
                    None => {
                        let err_msg = "There is no model available in the chat graphs.";

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        Err(LlamaCoreError::Operation(err_msg.into()))
                    }
                },
            }
        }
        None => {
            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    // update metadata
                    set_tensor_data_u8(graph, 1, config.as_bytes())
                }
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    }
}

fn reset_model_metadata(model_name: Option<&String>) -> Result<(), LlamaCoreError> {
    // get metadata
    let metadata = get_model_metadata(model_name)?;

    // update model with the original metadata
    update_model_metadata(model_name, &metadata)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextFullState {
    Message,
    Usage,
    Done,
    EndOfSequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    Usage,
    NoUsage,
    Done,
    EndOfSequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptTooLongState {
    Message,
    Usage,
    Done,
    EndOfSequence,
}

struct ChatStream {
    id: String,
    model: Option<String>,
    include_usage: bool,
    context_full_state: ContextFullState,
    prompt_too_long_state: PromptTooLongState,
    stream_state: StreamState,
    cache: Option<VecDeque<String>>,
}
impl ChatStream {
    fn new(
        model: Option<String>,
        id: String,
        include_usage: bool,
        cache: Option<Vec<String>>,
    ) -> Self {
        let stream_state = if include_usage {
            StreamState::Usage
        } else {
            StreamState::NoUsage
        };

        ChatStream {
            id,
            model,
            include_usage,
            context_full_state: ContextFullState::Message,
            prompt_too_long_state: PromptTooLongState::Message,
            stream_state,
            cache: cache.map(VecDeque::from),
        }
    }
}
impl Drop for ChatStream {
    fn drop(&mut self) {
        if self.cache.is_none() {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Clean up the context of the stream work environment.");

            match &self.model {
                Some(model_name) => {
                    match CHAT_GRAPHS.get() {
                        Some(chat_graphs) => match chat_graphs.lock() {
                            Ok(mut chat_graphs) => match chat_graphs.contains_key(model_name) {
                                true => {
                                    let graph = chat_graphs.get_mut(model_name).unwrap();

                                    if let Err(e) = graph.finish_single() {
                                        let err_msg = format!(
                                            "Failed to clean up the context. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        #[cfg(not(feature = "logging"))]
                                        println!(
                                            "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                            &err_msg
                                        );
                                    }
                                }
                                false => match chat_graphs.iter_mut().next() {
                                    Some((_, graph)) => {
                                        if let Err(e) = graph.finish_single() {
                                            let err_msg = format!(
                                                "Failed to clean up the context. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            #[cfg(not(feature = "logging"))]
                                            println!(
                                                "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                                &err_msg
                                            );
                                        }
                                    }
                                    None => {
                                        let err_msg =
                                            "There is no model available in the chat graphs.";

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        #[cfg(not(feature = "logging"))]
                                        println!(
                                            "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                            &err_msg
                                        );
                                    }
                                },
                            },
                            Err(e) => {
                                let err_msg =
                                    format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                #[cfg(not(feature = "logging"))]
                                println!(
                                "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                &err_msg
                            );
                            }
                        },
                        None => {
                            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            #[cfg(not(feature = "logging"))]
                            println!(
                                "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                &err_msg
                            );
                        }
                    };
                }
                None => {
                    match CHAT_GRAPHS.get() {
                        Some(chat_graphs) => match chat_graphs.lock() {
                            Ok(mut chat_graphs) => match chat_graphs.iter_mut().next() {
                                Some((_, graph)) => {
                                    if let Err(e) = graph.finish_single() {
                                        let err_msg = format!(
                                            "Failed to clean up the context. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        #[cfg(not(feature = "logging"))]
                                        println!(
                                        "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                        &err_msg
                                    );
                                    }
                                }
                                None => {
                                    let err_msg = "There is no model available in the chat graphs.";

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", err_msg);

                                    #[cfg(not(feature = "logging"))]
                                    println!(
                                    "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                    err_msg
                                );
                                }
                            },
                            Err(e) => {
                                let err_msg =
                                    format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                #[cfg(not(feature = "logging"))]
                                println!(
                                "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                &err_msg
                            );
                            }
                        },
                        None => {
                            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            #[cfg(not(feature = "logging"))]
                            println!(
                                "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                &err_msg
                            );
                        }
                    };
                }
            }

            #[cfg(feature = "logging")]
            info!(target: "stdout", "Cleanup done!");
        }
    }
}
impl futures::Stream for ChatStream {
    type Item = Result<String, LlamaCoreError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.cache.is_none() {
            let this = self.get_mut();
            let res = compute_stream(
                this.model.clone(),
                this.id.clone(),
                this.include_usage,
                &mut this.prompt_too_long_state,
                &mut this.context_full_state,
                &mut this.stream_state,
            );

            match res {
                Ok(x) => {
                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "next item: {}", &x);

                    if x != "[GGML] End of sequence" && !x.is_empty() {
                        Poll::Ready(Some(Ok(x)))
                    } else {
                        // stopped
                        Poll::Ready(None)
                    }
                }
                Err(e) => Poll::Ready(Some(Err(e))),
            }
        } else {
            let this = self.get_mut();

            let x = this.cache.as_mut().unwrap().pop_front();

            #[cfg(feature = "logging")]
            info!(target: "stdout", "Get the next item from the cache: {:?}", &x);

            match x {
                Some(x) => Poll::Ready(Some(Ok(x))),
                None => Poll::Ready(None),
            }
        }
    }
}

fn compute_stream(
    model_name: Option<String>,
    id: String,
    include_usage: bool,
    prompt_too_long_state: &mut PromptTooLongState,
    context_full_state: &mut ContextFullState,
    stream_state: &mut StreamState,
) -> Result<String, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute the chat stream chunk.");

    #[cfg(feature = "logging")]
    debug!(target: "stdout", "prompt_too_long_state: {:?}", *prompt_too_long_state);
    #[cfg(feature = "logging")]
    debug!(target: "stdout", "context_full_state: {:?}", *context_full_state);
    #[cfg(feature = "logging")]
    debug!(target: "stdout", "stream_state: {:?}", *stream_state);

    if *prompt_too_long_state == PromptTooLongState::EndOfSequence
        || *context_full_state == ContextFullState::EndOfSequence
        || *stream_state == StreamState::EndOfSequence
    {
        #[cfg(feature = "logging")]
        info!(target: "stdout", "Return the chat stream chunk!");

        return Ok("[GGML] End of sequence".to_string());
    }

    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let mut chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    // get graph
    let res = match &model_name {
        Some(model_name) => {
            match chat_graphs.contains_key(model_name) {
                true => {
                    let graph = chat_graphs.get_mut(model_name).unwrap();
                    // compute
                    match graph.compute_single() {
                        Ok(_) => {
                            #[cfg(feature = "logging")]
                            debug!(target: "stdout", "Compute the chat stream chunk successfully.");

                            match stream_state {
                                StreamState::Usage | StreamState::NoUsage => {
                                    // Retrieve the output
                                    let output_buffer =
                                        get_output_buffer_single(graph, OUTPUT_TENSOR)?;

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "retrieved the output buffer");

                                    // decode the output buffer to a utf8 string
                                    let output = match String::from_utf8(output_buffer.clone()) {
                                        Ok(token) => token,
                                        Err(_) => {
                                            let mutex = CACHED_UTF8_ENCODINGS
                                                .get_or_init(|| Mutex::new(Vec::new()));
                                            let mut cached_encodings = mutex.lock().map_err(|e| {
                                            let err_msg = format!(
                                                "Fail to acquire the lock of `UTF8_ENCODINGS`. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);


                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                            // cache the bytes for future decoding
                                            cached_encodings.extend_from_slice(&output_buffer[..]);

                                            match String::from_utf8(cached_encodings.to_vec()) {
                                                Ok(token) => {
                                                    // clear CACHED_UTF8_ENCODINGS
                                                    cached_encodings.clear();

                                                    token
                                                }
                                                Err(e) => {
                                                    // TODO This is a temp check. In case, infinite cached encodings happen.
                                                    if cached_encodings.len() > 4 {
                                                        let err_msg = format!("Fail to convert a vector of bytes to string. The length of the utf8 bytes exceeds 4. {}", e);

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "The cached buffer: {:?}", &cached_encodings[..]);

                                                        let token = String::from_utf8_lossy(
                                                            &cached_encodings,
                                                        )
                                                        .to_string();

                                                        // clear CACHED_UTF8_ENCODINGS
                                                        cached_encodings.clear();

                                                        token
                                                    } else {
                                                        let warn_msg = format!("Fail to convert a vector of bytes to string. {}", e);

                                                        #[cfg(feature = "logging")]
                                                        warn!(target: "stdout", "{}", &warn_msg);

                                                        String::from(" ")
                                                    }
                                                }
                                            }
                                        }
                                    };

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "decoded the output buffer");

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: Some(output),
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: None,
                                        }],
                                        usage: None,
                                    };

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "created chat completion chunk");

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                StreamState::Done => {
                                    *stream_state = StreamState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                StreamState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::EndOfSequence,
                        )) => {
                            #[cfg(feature = "logging")]
                            debug!(target: "stdout", "End of sequence");

                            match stream_state {
                                StreamState::Usage => {
                                    *stream_state = StreamState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "token_info: {} prompt tokens, {} completion tokens", token_info.prompt_tokens, token_info.completion_tokens);

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                StreamState::Done | StreamState::NoUsage => {
                                    *stream_state = StreamState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                StreamState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::ContextFull,
                        )) => {
                            #[cfg(feature = "logging")]
                            debug!(target: "stdout", "Context full");

                            match context_full_state {
                                ContextFullState::Message => {
                                    match include_usage {
                                        true => *context_full_state = ContextFullState::Usage,
                                        false => *context_full_state = ContextFullState::Done,
                                    }

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: Some(
                                                    "<|WASMEDGE-GGML-CONTEXT-FULL|>".to_string(),
                                                ),
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: Some(FinishReason::length),
                                        }],
                                        usage: None,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                ContextFullState::Usage => {
                                    *context_full_state = ContextFullState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                ContextFullState::Done => {
                                    *context_full_state = ContextFullState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                ContextFullState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::PromptTooLong,
                        )) => {
                            #[cfg(feature = "logging")]
                            debug!(target: "stdout", "Prompt too long");

                            match prompt_too_long_state {
                                PromptTooLongState::Message => {
                                    match include_usage {
                                        true => *prompt_too_long_state = PromptTooLongState::Usage,
                                        false => *prompt_too_long_state = PromptTooLongState::Done,
                                    }

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: None,
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: Some(FinishReason::length),
                                        }],
                                        usage: None,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                PromptTooLongState::Usage => {
                                    *prompt_too_long_state = PromptTooLongState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                PromptTooLongState::Done => {
                                    *prompt_too_long_state = PromptTooLongState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                PromptTooLongState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(e) => {
                            let err_msg =
                                format!("Failed to compute the chat completion. Reason: {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                                err_msg,
                            )))
                        }
                    }
                }
                false => {
                    match chat_graphs.iter_mut().next() {
                        Some((_, graph)) => {
                            // compute
                            match graph.compute_single() {
                                Ok(_) => {
                                    #[cfg(feature = "logging")]
                                    debug!(target: "stdout", "Compute the chat stream chunk successfully.");

                                    match stream_state {
                                        StreamState::Usage | StreamState::NoUsage => {
                                            // Retrieve the output
                                            let output_buffer =
                                                get_output_buffer_single(graph, OUTPUT_TENSOR)?;

                                            #[cfg(feature = "logging")]
                                            info!(target: "stdout", "retrieved the output buffer");

                                            // decode the output buffer to a utf8 string
                                            let output = match String::from_utf8(
                                                output_buffer.clone(),
                                            ) {
                                                Ok(token) => token,
                                                Err(_) => {
                                                    let mutex = CACHED_UTF8_ENCODINGS
                                                        .get_or_init(|| Mutex::new(Vec::new()));
                                                    let mut cached_encodings = mutex.lock().map_err(|e| {
                                            let err_msg = format!(
                                                "Fail to acquire the lock of `UTF8_ENCODINGS`. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);


                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                                    // cache the bytes for future decoding
                                                    cached_encodings
                                                        .extend_from_slice(&output_buffer[..]);

                                                    match String::from_utf8(
                                                        cached_encodings.to_vec(),
                                                    ) {
                                                        Ok(token) => {
                                                            // clear encodings
                                                            cached_encodings.clear();

                                                            token
                                                        }
                                                        Err(e) => {
                                                            // TODO This is a temp check. In case, infinite cached encodings happen.
                                                            if cached_encodings.len() > 4 {
                                                                let err_msg = format!("Fail to convert a vector of bytes to string. The length of the utf8 bytes exceeds 4. {}", e);

                                                                #[cfg(feature = "logging")]
                                                                error!(target: "stdout", "{}", &err_msg);

                                                                #[cfg(feature = "logging")]
                                                                error!(target: "stdout", "The cached buffer: {:?}", &cached_encodings[..]);

                                                                let token =
                                                                    String::from_utf8_lossy(
                                                                        &cached_encodings,
                                                                    )
                                                                    .to_string();

                                                                // clear CACHED_UTF8_ENCODINGS
                                                                cached_encodings.clear();

                                                                token
                                                            } else {
                                                                let warn_msg = format!("Fail to convert a vector of bytes to string. {}", e);

                                                                #[cfg(feature = "logging")]
                                                                warn!(target: "stdout", "{}", &warn_msg);

                                                                String::from(" ")
                                                            }
                                                        }
                                                    }
                                                }
                                            };

                                            #[cfg(feature = "logging")]
                                            info!(target: "stdout", "decoded the output buffer");

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![ChatCompletionChunkChoice {
                                                    index: 0,
                                                    delta: ChatCompletionChunkChoiceDelta {
                                                        role: ChatCompletionRole::Assistant,
                                                        content: Some(output),
                                                        tool_calls: vec![],
                                                    },
                                                    logprobs: None,
                                                    finish_reason: None,
                                                }],
                                                usage: None,
                                            };

                                            #[cfg(feature = "logging")]
                                            info!(target: "stdout", "created chat completion chunk");

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        StreamState::Done => {
                                            *stream_state = StreamState::EndOfSequence;

                                            Ok("data: [DONE]\n\n".to_string())
                                        }
                                        StreamState::EndOfSequence => {
                                            Ok("[GGML] End of sequence".to_string())
                                        }
                                    }
                                }
                                Err(wasmedge_wasi_nn::Error::BackendError(
                                    wasmedge_wasi_nn::BackendError::EndOfSequence,
                                )) => {
                                    #[cfg(feature = "logging")]
                                    debug!(target: "stdout", "End of sequence");

                                    match stream_state {
                                        StreamState::Usage => {
                                            *stream_state = StreamState::Done;

                                            // retrieve the number of prompt and completion tokens
                                            let token_info = get_token_info_by_graph(graph)?;

                                            let usage = Some(Usage {
                                                prompt_tokens: token_info.prompt_tokens,
                                                completion_tokens: token_info.completion_tokens,
                                                total_tokens: token_info.prompt_tokens
                                                    + token_info.completion_tokens,
                                            });

                                            #[cfg(feature = "logging")]
                                            info!(target: "stdout", "token_info: {} prompt tokens, {} completion tokens", token_info.prompt_tokens, token_info.completion_tokens);

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![],
                                                usage,
                                            };

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        StreamState::Done | StreamState::NoUsage => {
                                            *stream_state = StreamState::EndOfSequence;

                                            Ok("data: [DONE]\n\n".to_string())
                                        }
                                        StreamState::EndOfSequence => {
                                            Ok("[GGML] End of sequence".to_string())
                                        }
                                    }
                                }
                                Err(wasmedge_wasi_nn::Error::BackendError(
                                    wasmedge_wasi_nn::BackendError::ContextFull,
                                )) => {
                                    #[cfg(feature = "logging")]
                                    debug!(target: "stdout", "Context full");

                                    match context_full_state {
                                        ContextFullState::Message => {
                                            match include_usage {
                                                true => {
                                                    *context_full_state = ContextFullState::Usage
                                                }
                                                false => {
                                                    *context_full_state = ContextFullState::Done
                                                }
                                            }

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![ChatCompletionChunkChoice {
                                                    index: 0,
                                                    delta: ChatCompletionChunkChoiceDelta {
                                                        role: ChatCompletionRole::Assistant,
                                                        content: Some(
                                                            "<|WASMEDGE-GGML-CONTEXT-FULL|>"
                                                                .to_string(),
                                                        ),
                                                        tool_calls: vec![],
                                                    },
                                                    logprobs: None,
                                                    finish_reason: Some(FinishReason::length),
                                                }],
                                                usage: None,
                                            };

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        ContextFullState::Usage => {
                                            *context_full_state = ContextFullState::Done;

                                            // retrieve the number of prompt and completion tokens
                                            let token_info = get_token_info_by_graph(graph)?;

                                            let usage = Some(Usage {
                                                prompt_tokens: token_info.prompt_tokens,
                                                completion_tokens: token_info.completion_tokens,
                                                total_tokens: token_info.prompt_tokens
                                                    + token_info.completion_tokens,
                                            });

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![],
                                                usage,
                                            };

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        ContextFullState::Done => {
                                            *context_full_state = ContextFullState::EndOfSequence;

                                            Ok("data: [DONE]\n\n".to_string())
                                        }
                                        ContextFullState::EndOfSequence => {
                                            Ok("[GGML] End of sequence".to_string())
                                        }
                                    }
                                }
                                Err(wasmedge_wasi_nn::Error::BackendError(
                                    wasmedge_wasi_nn::BackendError::PromptTooLong,
                                )) => {
                                    #[cfg(feature = "logging")]
                                    debug!(target: "stdout", "Prompt too long");

                                    match prompt_too_long_state {
                                        PromptTooLongState::Message => {
                                            match include_usage {
                                                true => {
                                                    *prompt_too_long_state =
                                                        PromptTooLongState::Usage
                                                }
                                                false => {
                                                    *prompt_too_long_state =
                                                        PromptTooLongState::Done
                                                }
                                            }

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![ChatCompletionChunkChoice {
                                                    index: 0,
                                                    delta: ChatCompletionChunkChoiceDelta {
                                                        role: ChatCompletionRole::Assistant,
                                                        content: None,
                                                        tool_calls: vec![],
                                                    },
                                                    logprobs: None,
                                                    finish_reason: Some(FinishReason::length),
                                                }],
                                                usage: None,
                                            };

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        PromptTooLongState::Usage => {
                                            *prompt_too_long_state = PromptTooLongState::Done;

                                            // retrieve the number of prompt and completion tokens
                                            let token_info = get_token_info_by_graph(graph)?;

                                            let usage = Some(Usage {
                                                prompt_tokens: token_info.prompt_tokens,
                                                completion_tokens: token_info.completion_tokens,
                                                total_tokens: token_info.prompt_tokens
                                                    + token_info.completion_tokens,
                                            });

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![],
                                                usage,
                                            };

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        PromptTooLongState::Done => {
                                            *prompt_too_long_state =
                                                PromptTooLongState::EndOfSequence;

                                            Ok("data: [DONE]\n\n".to_string())
                                        }
                                        PromptTooLongState::EndOfSequence => {
                                            Ok("[GGML] End of sequence".to_string())
                                        }
                                    }
                                }
                                Err(e) => {
                                    let err_msg = format!(
                                        "Failed to compute the chat completion. Reason: {}",
                                        e
                                    );

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", &err_msg);

                                    Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                                        err_msg,
                                    )))
                                }
                            }
                        }
                        None => {
                            let err_msg = "There is no model available in the chat graphs.";

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            Err(LlamaCoreError::Operation(err_msg.into()))
                        }
                    }
                }
            }
        }
        None => {
            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    // compute
                    match graph.compute_single() {
                        Ok(_) => {
                            #[cfg(feature = "logging")]
                            debug!(target: "stdout", "Compute the chat stream chunk successfully.");

                            match stream_state {
                                StreamState::Usage | StreamState::NoUsage => {
                                    // Retrieve the output
                                    let output_buffer =
                                        get_output_buffer_single(graph, OUTPUT_TENSOR)?;

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "retrieved the output buffer");

                                    // decode the output buffer to a utf8 string
                                    let output = match String::from_utf8(output_buffer.clone()) {
                                        Ok(token) => token,
                                        Err(_) => {
                                            let mutex = CACHED_UTF8_ENCODINGS
                                                .get_or_init(|| Mutex::new(Vec::new()));
                                            let mut cached_encodings = mutex.lock().map_err(|e| {
                                            let err_msg = format!(
                                                "Fail to acquire the lock of `UTF8_ENCODINGS`. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                            cached_encodings.extend_from_slice(&output_buffer[..]);

                                            match String::from_utf8(cached_encodings.to_vec()) {
                                                Ok(token) => {
                                                    // clear encodings
                                                    cached_encodings.clear();

                                                    token
                                                }
                                                Err(e) => {
                                                    // TODO This is a temp check. In case, infinite cached encodings happen.
                                                    if cached_encodings.len() > 4 {
                                                        let err_msg = format!("Fail to convert a vector of bytes to string. The length of the utf8 bytes exceeds 4. {}", e);

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "The cached buffer: {:?}", &cached_encodings[..]);

                                                        let token = String::from_utf8_lossy(
                                                            &cached_encodings,
                                                        )
                                                        .to_string();

                                                        // clear CACHED_UTF8_ENCODINGS
                                                        cached_encodings.clear();

                                                        token
                                                    } else {
                                                        let warn_msg = format!("Fail to convert a vector of bytes to string. {}", e);

                                                        #[cfg(feature = "logging")]
                                                        warn!(target: "stdout", "{}", &warn_msg);

                                                        String::from(" ")
                                                    }
                                                }
                                            }
                                        }
                                    };

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "decoded the output buffer");

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: Some(output),
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: None,
                                        }],
                                        usage: None,
                                    };

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "created chat completion chunk");

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                StreamState::Done => {
                                    *stream_state = StreamState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                StreamState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::EndOfSequence,
                        )) => {
                            #[cfg(feature = "logging")]
                            debug!(target: "stdout", "End of sequence");

                            match stream_state {
                                StreamState::Usage => {
                                    *stream_state = StreamState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "token_info: {} prompt tokens, {} completion tokens", token_info.prompt_tokens, token_info.completion_tokens);

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                StreamState::Done | StreamState::NoUsage => {
                                    *stream_state = StreamState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                StreamState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::ContextFull,
                        )) => {
                            #[cfg(feature = "logging")]
                            debug!(target: "stdout", "Context full");

                            match context_full_state {
                                ContextFullState::Message => {
                                    match include_usage {
                                        true => *context_full_state = ContextFullState::Usage,
                                        false => *context_full_state = ContextFullState::Done,
                                    }

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: Some(
                                                    "<|WASMEDGE-GGML-CONTEXT-FULL|>".to_string(),
                                                ),
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: Some(FinishReason::length),
                                        }],
                                        usage: None,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                ContextFullState::Usage => {
                                    *context_full_state = ContextFullState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                ContextFullState::Done => {
                                    *context_full_state = ContextFullState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                ContextFullState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::PromptTooLong,
                        )) => {
                            #[cfg(feature = "logging")]
                            debug!(target: "stdout", "Prompt too long");

                            match prompt_too_long_state {
                                PromptTooLongState::Message => {
                                    match include_usage {
                                        true => *prompt_too_long_state = PromptTooLongState::Usage,
                                        false => *prompt_too_long_state = PromptTooLongState::Done,
                                    }

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: None,
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: Some(FinishReason::length),
                                        }],
                                        usage: None,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                PromptTooLongState::Usage => {
                                    *prompt_too_long_state = PromptTooLongState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                PromptTooLongState::Done => {
                                    *prompt_too_long_state = PromptTooLongState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                PromptTooLongState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(e) => {
                            let err_msg =
                                format!("Failed to compute the chat completion. Reason: {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                                err_msg,
                            )))
                        }
                    }
                }
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Return the chat stream chunk!");

    res
}

#[allow(dead_code)]
#[derive(Debug)]
struct ParseResult {
    raw: String,
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
}
