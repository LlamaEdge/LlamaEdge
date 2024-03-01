use crate::{
    error, print_log_begin_separator, print_log_end_separator, Graph, CTX_SIZE, GRAPH,
    MAX_BUFFER_SIZE, METADATA, UTF8_ENCODINGS,
};
use chat_prompts::{
    chat::{
        belle::HumanAssistantChatPrompt,
        llama::{CodeLlamaInstructPrompt, CodeLlamaSuperInstructPrompt, Llama2ChatPrompt},
        mistral::{MistralInstructPrompt, MistralLitePrompt},
        openchat::OpenChatPrompt,
        BuildChatPrompt, ChatPrompt,
    },
    PromptTemplateType,
};
use endpoints::{
    chat::{
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkChoiceDelta,
        ChatCompletionObject, ChatCompletionObjectChoice, ChatCompletionObjectMessage,
        ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionRole,
        ChatCompletionUserMessageContent, ContentPart,
    },
    common::{FinishReason, Usage},
};
use error::{BackendError, LlamaCoreError};
use futures::{
    future,
    stream::{self, TryStreamExt},
};

use serde_json::Value;
use std::{sync::Mutex, time::SystemTime};

/// Processes a chat-completion request and returns ChatCompletionChunk instances in stream.
pub async fn chat_completions_stream(
    chat_request: &mut ChatCompletionRequest,
    template_ty: PromptTemplateType,
    log_prompts: bool,
) -> Result<impl futures::TryStream<Ok = String, Error = LlamaCoreError>, LlamaCoreError> {
    // create ChatPrompt instance from the template type
    let template = create_prompt_template(template_ty);

    // build prompt
    let (prompt, avaible_completion_tokens) = build_prompt(&template, chat_request)
        .map_err(|e| LlamaCoreError::Operation(e.to_string()))?;

    if log_prompts {
        print_log_begin_separator("PROMPT", Some("*"), None);
        println!("\n{}", &prompt,);
        print_log_end_separator(Some("*"), None);
    }

    // update metadata
    update_metadata(&chat_request, avaible_completion_tokens).await?;

    let graph = GRAPH.get().ok_or(LlamaCoreError::Operation(String::from(
        "Fail to get the underlying value of `GRAPH`.",
    )))?;
    let mut graph = graph.lock().map_err(|e| {
        LlamaCoreError::Operation(format!(
            "Fail to acquire the lock of `GRAPH`. {}",
            e.to_string()
        ))
    })?;

    // set input
    let tensor_data = prompt.as_bytes().to_vec();
    if graph
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .is_err()
    {
        return Err(LlamaCoreError::Operation(String::from(
            "Fail to set input tensor",
        )));
    };

    let model = chat_request.model.clone().unwrap_or_default();

    let stop = {
        let metadata = METADATA
            .get()
            .ok_or(LlamaCoreError::Operation(String::from(
                "Fail to get the underlying value of `METADATA`.",
            )))?;
        metadata.reverse_prompt.clone()
    };
    let ref_stop = std::sync::Arc::new(stop);

    let mut one_more_run_then_stop = true;
    let stream = stream::repeat_with(move || {
        let reverse_prompt = ref_stop.clone();

        let graph = GRAPH.get().ok_or(LlamaCoreError::Operation(String::from(
            "Fail to get the underlying value of `GRAPH`.",
        )))?;
        let mut graph = graph.lock().map_err(|e| {
            LlamaCoreError::Operation(format!(
                "Fail to acquire the lock of `GRAPH`. {}",
                e.to_string()
            ))
        })?;

        // compute
        match graph.compute_single() {
            Ok(_) => {
                match one_more_run_then_stop {
                    true => {
                        // Retrieve the output.
                        let max_buffer_size = MAX_BUFFER_SIZE.get().ok_or(
                            LlamaCoreError::Operation(String::from(
                                "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
                            )),
                        )?;

                        let mut output_buffer = vec![0u8; *max_buffer_size];
                        let mut output_size = graph
                            .get_output_single(0, &mut output_buffer)
                            .map_err(|e| {
                                LlamaCoreError::Backend(BackendError::GetOutputSingle(format!(
                                    "Fail to get output tensor: {msg}",
                                    msg = e.to_string()
                                )))
                            })?;
                        output_size = std::cmp::min(*max_buffer_size, output_size);

                        // decode the output buffer to a utf8 string
                        let output = match String::from_utf8(output_buffer[..output_size].to_vec())
                        {
                            Ok(token) => token,
                            Err(_) => {
                                let mutex = UTF8_ENCODINGS.get_or_init(|| Mutex::new(Vec::new()));
                                let mut cached_encodings = mutex.lock().map_err(|e| {
                                    LlamaCoreError::Operation(format!(
                                        "Fail to acquire the lock of `UTF8_ENCODINGS`. {}",
                                        e.to_string()
                                    ))
                                })?;

                                cached_encodings.extend_from_slice(&output_buffer[..output_size]);

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

                        if let Some(stop) = &*reverse_prompt.clone() {
                            if output == *stop {
                                let created = SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map_err(|e| {
                                        LlamaCoreError::Operation(format!(
                                            "Failed to get the current time. {}",
                                            e.to_string()
                                        ))
                                    })?;

                                let chat_completion_chunk = ChatCompletionChunk {
                                    id: "chatcmpl-123".to_string(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: created.as_secs(),
                                    model: model.clone(),
                                    system_fingerprint: "fp_44709d6fcb".to_string(),
                                    choices: vec![ChatCompletionChunkChoice {
                                        index: 0,
                                        delta: ChatCompletionChunkChoiceDelta {
                                            role: Some(ChatCompletionRole::Assistant),
                                            content: Some("data: [DONE]".to_string()),
                                            function_call: None,
                                            tool_calls: None,
                                        },
                                        logprobs: None,
                                        finish_reason: Some(FinishReason::stop),
                                    }],
                                };

                                one_more_run_then_stop = false;

                                // serialize chat completion chunk
                                let chunk =
                                    serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                        LlamaCoreError::Operation(format!(
                                            "Failed to serialize chat completion chunk. {}",
                                            e.to_string()
                                        ))
                                    })?;

                                return Ok(chunk);
                            }
                        }

                        let created = SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_err(|e| {
                                LlamaCoreError::Operation(format!(
                                    "Failed to get the current time. {}",
                                    e.to_string()
                                ))
                            })?;

                        let chat_completion_chunk = ChatCompletionChunk {
                            id: "chatcmpl-123".to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: created.as_secs(),
                            model: model.clone(),
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
                        };

                        // serialize chat completion chunk
                        let chunk = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                            LlamaCoreError::Operation(format!(
                                "Failed to serialize chat completion chunk. {}",
                                e.to_string()
                            ))
                        })?;

                        Ok(chunk)
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
            Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::EndOfSequence)) => {
                match one_more_run_then_stop {
                    true => {
                        let created = SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_err(|e| {
                                LlamaCoreError::Operation(format!(
                                    "Failed to get the current time. {}",
                                    e.to_string()
                                ))
                            })?;

                        let chat_completion_chunk = ChatCompletionChunk {
                            id: "chatcmpl-123".to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: created.as_secs(),
                            model: model.clone(),
                            system_fingerprint: "fp_44709d6fcb".to_string(),
                            choices: vec![ChatCompletionChunkChoice {
                                index: 0,
                                delta: ChatCompletionChunkChoiceDelta {
                                    role: Some(ChatCompletionRole::Assistant),
                                    content: Some("data: [DONE]".to_string()),
                                    function_call: None,
                                    tool_calls: None,
                                },
                                logprobs: None,
                                finish_reason: Some(FinishReason::stop),
                            }],
                        };

                        one_more_run_then_stop = false;

                        // serialize chat completion chunk
                        let chunk = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                            LlamaCoreError::Operation(format!(
                                "Failed to serialize chat completion chunk. {}",
                                e.to_string()
                            ))
                        })?;

                        Ok(chunk)
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
            Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::ContextFull)) => {
                match one_more_run_then_stop {
                    true => {
                        let created = SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_err(|e| {
                                LlamaCoreError::Operation(format!(
                                    "Failed to get the current time. {}",
                                    e.to_string()
                                ))
                            })?;

                        let chat_completion_chunk = ChatCompletionChunk {
                            id: "chatcmpl-123".to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: created.as_secs(),
                            model: model.clone(),
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
                        };

                        one_more_run_then_stop = false;

                        // serialize chat completion chunk
                        let chunk = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                            LlamaCoreError::Operation(format!(
                                "Failed to serialize chat completion chunk. {}",
                                e.to_string()
                            ))
                        })?;

                        Ok(chunk)
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
            Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::PromptTooLong)) => {
                match one_more_run_then_stop {
                    true => {
                        let created = SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_err(|e| {
                                LlamaCoreError::Operation(format!(
                                    "Failed to get the current time. {}",
                                    e.to_string()
                                ))
                            })?;

                        let chat_completion_chunk = ChatCompletionChunk {
                            id: "chatcmpl-123".to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: created.as_secs(),
                            model: model.clone(),
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
                        };

                        one_more_run_then_stop = false;

                        // serialize chat completion chunk
                        let chunk = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                            LlamaCoreError::Operation(format!(
                                "Failed to serialize chat completion chunk. {}",
                                e.to_string()
                            ))
                        })?;

                        Ok(chunk)
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
                return Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                    e.to_string(),
                )));
            }
        }
    })
    .try_take_while(|x| future::ready(Ok(x != "[GGML] End of sequence" && x != "")));

    Ok(stream)
}

/// Processes a chat-completion request and returns a ChatCompletionObject instance.
pub async fn chat_completions(
    chat_request: &mut ChatCompletionRequest,
    template_ty: PromptTemplateType,
    log_prompts: bool,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    // create ChatPrompt instance from the template type
    let template = create_prompt_template(template_ty);

    // build prompt
    let (prompt, avaible_completion_tokens) =
        build_prompt(&template, chat_request).map_err(|e| {
            LlamaCoreError::Operation(format!("Failed to build prompt. {}", e.to_string()))
        })?;

    if log_prompts {
        print_log_begin_separator("PROMPT", Some("*"), None);
        println!("\n{}", &prompt,);
        print_log_end_separator(Some("*"), None);
    }

    // update metadata
    update_metadata(&chat_request, avaible_completion_tokens).await?;

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

    // set input
    let tensor_data = prompt.as_bytes().to_vec();
    graph
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .map_err(|e| LlamaCoreError::Backend(BackendError::SetInput(e.to_string())))?;

    match graph.compute() {
        Ok(_) => {
            // Retrieve the output.
            let max_buffer_size =
                MAX_BUFFER_SIZE
                    .get()
                    .ok_or(LlamaCoreError::Operation(String::from(
                        "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
                    )))?;
            let mut output_buffer = vec![0u8; *max_buffer_size];
            let mut output_size: usize = graph.get_output(0, &mut output_buffer).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to get output tensor: {msg}",
                    msg = e.to_string()
                ))
            })?;
            output_size = std::cmp::min(*max_buffer_size, output_size);

            // convert inference result to string
            let output = std::str::from_utf8(&output_buffer[..output_size]).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e.to_string()
                ))
            })?;

            // post-process
            let message = post_process(&output, template_ty).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to post-process the output. {}",
                    e.to_string()
                ))
            })?;

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info(&graph).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to get the number of prompt and completion tokens. {}",
                    e.to_string()
                ))
            })?;
            if log_prompts {
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
                    LlamaCoreError::Operation(format!(
                        "Failed to get the current time. {}",
                        e.to_string()
                    ))
                })?;

            // create ChatCompletionResponse
            Ok(ChatCompletionObject {
                id: uuid::Uuid::new_v4().to_string(),
                object: String::from("chat.completion"),
                created: created.as_secs(),
                model: chat_request.model.clone().unwrap_or_default(),
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
        Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::ContextFull)) => {
            // Retrieve the output.
            let max_buffer_size =
                MAX_BUFFER_SIZE
                    .get()
                    .ok_or(LlamaCoreError::Operation(String::from(
                        "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
                    )))?;
            let mut output_buffer = vec![0u8; *max_buffer_size];
            let mut output_size = graph.get_output(0, &mut output_buffer).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to get output tensor: {msg}",
                    msg = e.to_string()
                ))
            })?;
            output_size = std::cmp::min(*max_buffer_size, output_size);

            // convert inference result to string
            let output = std::str::from_utf8(&output_buffer[..output_size]).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e.to_string()
                ))
            })?;

            // post-process
            let message = post_process(&output, template_ty).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to post-process the output. {}",
                    e.to_string()
                ))
            })?;

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info(&graph).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to get the number of prompt and completion tokens. {}",
                    e.to_string()
                ))
            })?;
            if log_prompts {
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
                    LlamaCoreError::Operation(format!(
                        "Failed to get the current time. {}",
                        e.to_string()
                    ))
                })?;

            // create ChatCompletionResponse
            Ok(ChatCompletionObject {
                id: uuid::Uuid::new_v4().to_string(),
                object: String::from("chat.completion"),
                created: created.as_secs(),
                model: chat_request.model.clone().unwrap_or_default(),
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
        Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::PromptTooLong)) => {
            println!("\n\n[WARNING] The prompt is too long. Please reduce the length of your input and try again.\n");

            // Retrieve the output.
            let max_buffer_size =
                MAX_BUFFER_SIZE
                    .get()
                    .ok_or(LlamaCoreError::Operation(String::from(
                        "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
                    )))?;
            let mut output_buffer = vec![0u8; *max_buffer_size];
            let mut output_size = graph.get_output(0, &mut output_buffer).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to get output tensor: {msg}",
                    msg = e.to_string()
                ))
            })?;
            output_size = std::cmp::min(*max_buffer_size, output_size);

            // convert inference result to string
            let output = std::str::from_utf8(&output_buffer[..output_size]).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e.to_string()
                ))
            })?;

            // post-process
            let message = post_process(output, template_ty).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to post-process the output. {}",
                    e.to_string()
                ))
            })?;

            // retrieve the number of prompt and completion token
            let token_info = get_token_info(&graph).map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Failed to get the number of prompt and completion tokens. {}",
                    e.to_string()
                ))
            })?;
            if log_prompts {
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
                    LlamaCoreError::Operation(format!(
                        "Failed to get the current time. {}",
                        e.to_string()
                    ))
                })?;

            // create ChatCompletionResponse
            Ok(ChatCompletionObject {
                id: uuid::Uuid::new_v4().to_string(),
                object: String::from("chat.completion"),
                created: created.as_secs(),
                model: chat_request.model.clone().unwrap_or_default(),
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

async fn update_metadata(
    chat_request: &ChatCompletionRequest,
    available_completion_tokens: u64,
) -> Result<(), LlamaCoreError> {
    let mut should_update = false;
    let mut metadata = match METADATA.get() {
        Some(metadata) => metadata.clone(),
        None => {
            return Err(LlamaCoreError::Operation(String::from(
                "Fail to get the underlying value of `METADATA`.",
            )));
        }
    };

    // check if necessary to update `image`
    if let Some(ChatCompletionRequestMessage::User(user_message)) = chat_request.messages.last() {
        if let ChatCompletionUserMessageContent::Parts(parts) = user_message.content() {
            for part in parts {
                if let ContentPart::Image(image) = part {
                    let image = image.image();
                    match image.is_url() {
                        true => {
                            // update metadata image
                            let img = download_image(&image.url).await?;

                            metadata.image = Some(img);

                            if !should_update {
                                should_update = true;
                            }

                            // todo: now only support a single image
                            break;
                        }
                        false => {
                            return Err(LlamaCoreError::Operation(String::from(
                                "Base64 image is not supported yet.",
                            )));
                        }
                    }
                }
            }
        }
    }

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
    } else {
        if metadata.n_predict > available_completion_tokens {
            // update n_predict
            metadata.n_predict = available_completion_tokens;

            if !should_update {
                should_update = true;
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

    if should_update {
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
            return Err(LlamaCoreError::Operation(String::from(
                "Fail to update metadata",
            )));
        }
    }

    Ok(())
}

fn create_prompt_template(template_ty: PromptTemplateType) -> ChatPrompt {
    match template_ty {
        PromptTemplateType::Llama2Chat => ChatPrompt::Llama2ChatPrompt(Llama2ChatPrompt::default()),
        PromptTemplateType::MistralInstruct => {
            ChatPrompt::MistralInstructPrompt(MistralInstructPrompt::default())
        }
        PromptTemplateType::MistralLite => {
            ChatPrompt::MistralLitePrompt(MistralLitePrompt::default())
        }
        PromptTemplateType::OpenChat => ChatPrompt::OpenChatPrompt(OpenChatPrompt::default()),
        PromptTemplateType::CodeLlama => {
            ChatPrompt::CodeLlamaInstructPrompt(CodeLlamaInstructPrompt::default())
        }
        PromptTemplateType::CodeLlamaSuper => {
            ChatPrompt::CodeLlamaSuperInstructPrompt(CodeLlamaSuperInstructPrompt::default())
        }
        PromptTemplateType::HumanAssistant => {
            ChatPrompt::HumanAssistantChatPrompt(HumanAssistantChatPrompt::default())
        }
        PromptTemplateType::VicunaChat => {
            ChatPrompt::VicunaChatPrompt(chat_prompts::chat::vicuna::VicunaChatPrompt::default())
        }
        PromptTemplateType::Vicuna11Chat => {
            ChatPrompt::Vicuna11ChatPrompt(chat_prompts::chat::vicuna::Vicuna11ChatPrompt::default())
        }
        PromptTemplateType::VicunaLlava => {
            ChatPrompt::VicunaLlavaPrompt(chat_prompts::chat::vicuna::VicunaLlavaPrompt::default())
        }
        PromptTemplateType::YiLlava => {
            ChatPrompt::YiLlavaPrompt(chat_prompts::chat::vicuna::YiLlavaPrompt::default())
        }
        PromptTemplateType::ChatML => {
            ChatPrompt::ChatMLPrompt(chat_prompts::chat::chatml::ChatMLPrompt::default())
        }
        PromptTemplateType::Baichuan2 => ChatPrompt::Baichuan2ChatPrompt(
            chat_prompts::chat::baichuan::Baichuan2ChatPrompt::default(),
        ),
        PromptTemplateType::WizardCoder => {
            ChatPrompt::WizardCoderPrompt(chat_prompts::chat::wizard::WizardCoderPrompt::default())
        }
        PromptTemplateType::Zephyr => {
            ChatPrompt::ZephyrChatPrompt(chat_prompts::chat::zephyr::ZephyrChatPrompt::default())
        }
        PromptTemplateType::StableLMZephyr => ChatPrompt::StableLMZephyrChatPrompt(
            chat_prompts::chat::zephyr::StableLMZephyrChatPrompt::default(),
        ),
        PromptTemplateType::IntelNeural => {
            ChatPrompt::NeuralChatPrompt(chat_prompts::chat::intel::NeuralChatPrompt::default())
        }
        PromptTemplateType::DeepseekChat => ChatPrompt::DeepseekChatPrompt(
            chat_prompts::chat::deepseek::DeepseekChatPrompt::default(),
        ),
        PromptTemplateType::DeepseekCoder => ChatPrompt::DeepseekCoderPrompt(
            chat_prompts::chat::deepseek::DeepseekCoderPrompt::default(),
        ),
        PromptTemplateType::SolarInstruct => ChatPrompt::SolarInstructPrompt(
            chat_prompts::chat::solar::SolarInstructPrompt::default(),
        ),
        PromptTemplateType::Phi2Chat => {
            ChatPrompt::Phi2ChatPrompt(chat_prompts::chat::phi::Phi2ChatPrompt::default())
        }
        PromptTemplateType::Phi2Instruct => {
            ChatPrompt::Phi2InstructPrompt(chat_prompts::chat::phi::Phi2InstructPrompt::default())
        }
        PromptTemplateType::GemmaInstruct => ChatPrompt::GemmaInstructPrompt(
            chat_prompts::chat::gemma::GemmaInstructPrompt::default(),
        ),
    }
}

fn post_process(
    output: impl AsRef<str>,
    template_ty: PromptTemplateType,
) -> Result<String, String> {
    let output = if template_ty == PromptTemplateType::Baichuan2 {
        if output.as_ref().contains("用户:") {
            output.as_ref().trim_end_matches("用户:").trim().to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if template_ty == PromptTemplateType::OpenChat {
        if output.as_ref().contains("<|end_of_turn|>") {
            output
                .as_ref()
                .trim_end_matches("<|end_of_turn|>")
                .trim()
                .to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if template_ty == PromptTemplateType::ChatML {
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
    } else if template_ty == PromptTemplateType::Zephyr
        || template_ty == PromptTemplateType::MistralLite
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
    } else if template_ty == PromptTemplateType::DeepseekChat {
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
    } else if template_ty == PromptTemplateType::HumanAssistant {
        if output.as_ref().contains("Human:") {
            output.as_ref().trim_end_matches("Human:").trim().to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if template_ty == PromptTemplateType::SolarInstruct {
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
    } else {
        output.as_ref().trim().to_owned()
    };

    Ok(output)
}

fn build_prompt(
    template: &ChatPrompt,
    chat_request: &mut ChatCompletionRequest,
) -> Result<(String, u64), String> {
    let graph = match GRAPH.get() {
        Some(graph) => graph,
        None => {
            return Err(String::from("Fail to get the underlying value of `GRAPH`."));
        }
    };
    let mut graph = match graph.lock() {
        Ok(graph) => graph,
        Err(e) => {
            return Err(format!(
                "Fail to acquire the lock of `GRAPH`. {}",
                e.to_string()
            ));
        }
    };

    let ctx_size = match CTX_SIZE.get() {
        Some(ctx_size) => *ctx_size as u64,
        None => {
            return Err(String::from(
                "Fail to get the underlying value of `CTX_SIZE`.",
            ));
        }
    };

    let max_prompt_tokens = ctx_size * 4 / 5;

    loop {
        // build prompt
        let prompt = match template.build(&mut chat_request.messages) {
            Ok(prompt) => prompt,
            Err(e) => {
                return Err(format!(
                    "Fail to build chat prompts: {msg}",
                    msg = e.to_string()
                ))
            }
        };

        // read input tensor
        let tensor_data = prompt.trim().as_bytes().to_vec();
        if graph
            .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
            .is_err()
        {
            return Err(String::from("Fail to set input tensor"));
        };

        // Retrieve the number of prompt tokens.
        let max_input_size = match MAX_BUFFER_SIZE.get() {
            Some(max_input_size) => *max_input_size,
            None => {
                return Err(String::from(
                    "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
                ));
            }
        };
        let mut input_buffer = vec![0u8; max_input_size];
        let mut input_size = match graph.get_output(1, &mut input_buffer) {
            Ok(size) => size,
            Err(e) => {
                return Err(format!(
                    "Fail to get token info: {msg}",
                    msg = e.to_string()
                ));
            }
        };
        input_size = std::cmp::min(max_input_size, input_size);
        let token_info: Value = match serde_json::from_slice(&input_buffer[..input_size]) {
            Ok(token_info) => token_info,
            Err(e) => {
                return Err(format!(
                    "Fail to deserialize token info: {msg}",
                    msg = e.to_string()
                ));
            }
        };
        let prompt_tokens = match token_info["input_tokens"].as_u64() {
            Some(prompt_tokens) => prompt_tokens,
            None => {
                return Err(String::from("Fail to get `input_tokens`."));
            }
        };

        match prompt_tokens > max_prompt_tokens {
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

pub(crate) fn get_token_info(graph: &Graph) -> Result<TokenInfo, String> {
    let max_output_size = match MAX_BUFFER_SIZE.get() {
        Some(max_output_size) => *max_output_size,
        None => {
            return Err(String::from(
                "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
            ));
        }
    };
    let mut output_buffer = vec![0u8; max_output_size];
    let mut output_size = match graph.get_output(1, &mut output_buffer) {
        Ok(size) => size,
        Err(e) => {
            return Err(format!(
                "Fail to get token info: {msg}",
                msg = e.to_string()
            ));
        }
    };
    output_size = std::cmp::min(max_output_size, output_size);
    let token_info: Value = match serde_json::from_slice(&output_buffer[..output_size]) {
        Ok(token_info) => token_info,
        Err(e) => {
            return Err(format!(
                "Fail to deserialize token info: {msg}",
                msg = e.to_string()
            ));
        }
    };

    let prompt_tokens = match token_info["input_tokens"].as_u64() {
        Some(prompt_tokens) => prompt_tokens,
        None => {
            return Err(String::from("Fail to convert `input_tokens` to u64."));
        }
    };
    let completion_tokens = match token_info["output_tokens"].as_u64() {
        Some(completion_tokens) => completion_tokens,
        None => {
            return Err(String::from("Fail to convert `output_tokens` to u64."));
        }
    };
    Ok(TokenInfo {
        prompt_tokens,
        completion_tokens,
    })
}

#[derive(Debug)]

pub(crate) struct TokenInfo {
    pub(crate) prompt_tokens: u64,
    pub(crate) completion_tokens: u64,
}

/// Downloads an image from the given URL and returns the file name.
async fn download_image(image_url: impl AsRef<str>) -> Result<String, LlamaCoreError> {
    let image_url = image_url.as_ref();
    // println!("[DEBUG] image_url: {}", image_url);
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
