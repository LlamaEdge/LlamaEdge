//! Define APIs for chat completion.

use crate::{
    error, running_mode,
    utils::{
        gen_chat_id, get_output_buffer, get_output_buffer_single, get_token_info_by_graph,
        get_token_info_by_graph_name, set_tensor_data_u8,
    },
    Graph, Metadata, RunningMode, CACHED_UTF8_ENCODINGS, CHAT_GRAPHS, OUTPUT_TENSOR,
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
#[cfg(feature = "https")]
use futures::StreamExt;
use std::{
    pin::Pin,
    sync::Mutex,
    task::{Context, Poll},
    time::SystemTime,
};

/// Processes a chat-completion request and returns ChatCompletionChunk instances in stream.
pub async fn chat_completions_stream(
    chat_request: &mut ChatCompletionRequest,
) -> Result<impl futures::TryStream<Ok = String, Error = LlamaCoreError>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "llama_core", "Process chat completion request in stream mode.");

    let running_mode = running_mode()?;
    if running_mode == RunningMode::Embeddings {
        let err_msg = format!(
            "The chat completion is not supported in the {} mode.",
            running_mode
        );

        #[cfg(feature = "logging")]
        error!(target: "llama_core", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    let model_name = chat_request.model.clone();
    let id = match &chat_request.user {
        Some(id) => id.clone(),
        None => gen_chat_id(),
    };

    #[cfg(feature = "logging")]
    info!(target: "llama_core", "user: {}", &id);

    // parse the `include_usage` option
    let include_usage = match chat_request.stream_options {
        Some(ref stream_options) => stream_options.include_usage.unwrap_or_default(),
        None => false,
    };

    #[cfg(feature = "logging")]
    info!(target: "llama_core", "include_usage: {}", include_usage);

    // update metadata
    let mut metadata = check_model_metadata(chat_request).await?;

    // build prompt
    let (prompt, avaible_completion_tokens) = build_prompt(model_name.as_ref(), chat_request)?;

    #[cfg(feature = "logging")]
    info!(target: "llama_core", "prompt: {}, avaible_completion_tokens: {}", &prompt, avaible_completion_tokens);

    // update metadata n_predict
    update_n_predict(chat_request, &mut metadata, avaible_completion_tokens).await?;

    // set prompt
    set_prompt(chat_request.model.as_ref(), &prompt)?;

    // create chat stream
    let stream = ChatStream::new(model_name, id, include_usage);

    #[cfg(feature = "logging")]
    info!(target: "llama_core", "End of the chat completion stream.");

    Ok(stream)
}

/// Processes a chat-completion request and returns a ChatCompletionObject instance.
pub async fn chat_completions(
    chat_request: &mut ChatCompletionRequest,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "llama_core", "Processing chat completion request in non-stream mode.");

    let running_mode = running_mode()?;
    if running_mode == RunningMode::Embeddings {
        let err_msg = format!(
            "The chat completion is not supported in the {} mode.",
            running_mode
        );

        #[cfg(feature = "logging")]
        error!(target: "llama_core", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    let model_name = chat_request.model.clone();
    let id = match &chat_request.user {
        Some(id) => id.clone(),
        None => gen_chat_id(),
    };

    #[cfg(feature = "logging")]
    info!(target: "llama_core", "user: {}", &id);

    // update metadata
    let mut metadata = check_model_metadata(chat_request).await?;

    // build prompt
    let (prompt, avaible_completion_tokens) = build_prompt(model_name.as_ref(), chat_request)?;

    #[cfg(feature = "logging")]
    {
        info!(target: "llama_core", "prompt:\n{}", &prompt);
        info!(target: "llama_core", "avaible_completion_tokens: {}", avaible_completion_tokens);
    }

    // update metadata n_predict
    update_n_predict(chat_request, &mut metadata, avaible_completion_tokens).await?;

    // feed the prompt to the model
    set_prompt(model_name.as_ref(), &prompt)?;

    // compute
    let res = compute(model_name.as_ref(), id);

    #[cfg(feature = "logging")]
    info!(target: "llama_core", "End of the chat completion.");

    res
}

fn compute(
    model_name: Option<&String>,
    id: impl Into<String>,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "llama_core", "Compute chat completion.");

    match model_name {
        Some(model_name) => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match chat_graphs.get_mut(model_name) {
                Some(graph) => compute_by_graph(graph, id),
                None => {
                    let err_msg = format!(
                        "The model `{}` does not exist in the chat graphs.",
                        &model_name
                    );

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        None => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => compute_by_graph(graph, id),
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    }
}

fn compute_by_graph(
    graph: &mut Graph,
    id: impl Into<String>,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "llama_core", "Compute chat completion by the model named {}.", graph.name());

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
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                LlamaCoreError::Operation(format!("Failed to post-process the output. {}", e))
            })?;

            #[cfg(feature = "logging")]
            info!(target: "llama_core", "post-processed generation: {}", &message);

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "llama_core", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

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
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                let err_msg = format!("Failed to post-process the output. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "llama_core", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

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
            #[cfg(feature = "logging")]
            warn!(target: "llama_core", "The prompt is too long. Please reduce the length of your input and try again.");

            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                let err_msg = format!("Failed to post-process the output. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // retrieve the number of prompt and completion token
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "llama_core", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

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
        Err(e) => {
            let err_msg = format!("Failed to compute the chat completion. Reason: {}", e);

            #[cfg(feature = "logging")]
            error!(target: "llama_core", "{}", &err_msg);

            Err(LlamaCoreError::Backend(BackendError::Compute(err_msg)))
        }
    }
}

async fn check_model_metadata(
    chat_request: &ChatCompletionRequest,
) -> Result<Metadata, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "llama_core", "Check model metadata.");

    let mut should_update = false;
    let mut metadata = get_model_metadata(chat_request.model.as_ref())?;

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
        update_model_metadata(chat_request.model.as_ref(), &metadata)?;
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

        #[cfg(feature = "logging")]
        info!(target: "llama_core", "n_predict: current: {}, new: {}", metadata.n_predict, max_completion_tokens);

        // update n_predict
        metadata.n_predict = max_completion_tokens;

        if !should_update {
            should_update = true;
        }
    } else if metadata.n_predict > available_completion_tokens {
        #[cfg(feature = "logging")]
        info!(target: "llama_core", "n_predict: current: {}, new: {}", metadata.n_predict, available_completion_tokens);

        // update n_predict
        metadata.n_predict = available_completion_tokens;

        if !should_update {
            should_update = true;
        }
    }

    if should_update {
        // update the target graph with the new metadata
        update_model_metadata(chat_request.model.as_ref(), metadata)?;
    }

    Ok(())
}

fn post_process(
    output: impl AsRef<str>,
    template_ty: &PromptTemplateType,
) -> Result<String, String> {
    #[cfg(feature = "logging")]
    info!(target: "llama_core", "Post-process the generated output.");

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
    #[cfg(feature = "logging")]
    info!(target: "llama_core", "Build the chat prompt from the chat messages.");

    let metadata = get_model_metadata(model_name)?;
    let ctx_size = metadata.ctx_size as u64;
    let chat_prompt = ChatPrompt::from(metadata.prompt_template);

    // compute max prompt tokens
    let max_prompt_tokens = ctx_size * 4 / 5;

    loop {
        // build prompt
        let prompt = match chat_prompt.build(&mut chat_request.messages) {
            Ok(prompt) => prompt,
            Err(e) => {
                let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                return Err(LlamaCoreError::Operation(err_msg));
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
                            #[cfg(feature = "logging")]
                            info!(target: "llama_core", "prompt: {}", &prompt);

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
                            #[cfg(feature = "logging")]
                            info!(target: "llama_core", "prompt: {}", &prompt);

                            return Ok((prompt, ctx_size - max_prompt_tokens));
                        }
                    }
                    _ => {
                        let err_msg = format!(
                            "Found a unsupported chat message role: {:?}",
                            chat_request.messages[0].role()
                        );

                        #[cfg(feature = "logging")]
                        error!(target: "llama_core", "{}", &err_msg);

                        panic!("{}", err_msg)
                    }
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
    #[cfg(feature = "logging")]
    info!(target: "llama_core", "Download image from the URL.");

    let image_url = image_url.as_ref();
    let url = reqwest::Url::parse(image_url).map_err(|e| {
        let err_msg = format!("Fail to parse the image URL: {}. Reason: {}", image_url, e);

        #[cfg(feature = "logging")]
        error!(target: "llama_core", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let response = reqwest::get(url).await.map_err(|e| {
        let err_msg = format!(
            "Fail to download the image from the URL: {}. Reason: {}",
            image_url, e
        );

        #[cfg(feature = "logging")]
        error!(target: "llama_core", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

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

    let mut dest = std::fs::File::create(&fname).map_err(|e| {
        let err_msg = format!(
            "Fail to create the file to save the image: {}. Reason: {}",
            &fname, e
        );

        #[cfg(feature = "logging")]
        error!(target: "llama_core", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let mut content = response.bytes_stream();
    while let Ok(item) = content.next().await.unwrap() {
        std::io::copy(&mut item.as_ref(), &mut dest).map_err(|e| {
            let err_msg = format!(
                "Fail to write the image content to the file: {}. Reason: {}",
                &fname, e
            );

            #[cfg(feature = "logging")]
            error!(target: "llama_core", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;
    }

    #[cfg(feature = "logging")]
    info!(target: "llama_core", "The image is downloaded successfully.");

    Ok(fname)
}

fn set_prompt(model_name: Option<&String>, prompt: impl AsRef<str>) -> Result<(), LlamaCoreError> {
    match model_name {
        Some(model_name) => {
            #[cfg(feature = "logging")]
            info!(target: "llama_core", "Set prompt to the chat model named {}.", model_name);

            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = format!("Fail to get the underlying value of `CHAT_GRAPHS` while trying to set prompt to the model named {}.", model_name);

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg));
                }
            };

            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS` while trying to set prompt to the model named {}. Reason: {}", model_name, e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match chat_graphs.get_mut(model_name) {
                Some(graph) => {
                    let tensor_data = prompt.as_ref().as_bytes().to_vec();
                    set_tensor_data_u8(graph, 0, &tensor_data)
                }
                None => {
                    let err_msg = format!(
                        "The model `{}` does not exist in the chat graphs while trying to set prompt.",
                        &model_name
                    );

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        None => {
            #[cfg(feature = "logging")]
            info!(target: "llama_core", "Set prompt to the default chat model.");

            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS` while trying to set prompt to the default model.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`while trying to set prompt to the default model. Reason: {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    let tensor_data = prompt.as_ref().as_bytes().to_vec();
                    set_tensor_data_u8(graph, 0, &tensor_data)
                }
                None => {
                    let err_msg = "There is no model available in the chat graphs while trying to set prompt to the default model.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
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

/// Get a copy of the metadata of the model.
fn get_model_metadata(model_name: Option<&String>) -> Result<Metadata, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "llama_core", "Get the model metadata.");

    match model_name {
        Some(model_name) => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match chat_graphs.get(model_name) {
                Some(graph) => Ok(graph.metadata.clone()),
                None => {
                    let err_msg = format!(
                        "The model `{}` does not exist in the chat graphs.",
                        &model_name
                    );

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        None => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match chat_graphs.iter().next() {
                Some((_, graph)) => Ok(graph.metadata.clone()),
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    }
}

fn update_model_metadata(
    model_name: Option<&String>,
    metadata: &Metadata,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "llama_core", "Update the model metadata.");

    let config = match serde_json::to_string(metadata) {
        Ok(config) => config,
        Err(e) => {
            let err_msg = format!("Fail to serialize metadata to a JSON string. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "llama_core", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg));
        }
    };

    match model_name {
        Some(model_name) => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. Reason: {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match chat_graphs.get_mut(model_name) {
                Some(graph) => {
                    // update metadata
                    set_tensor_data_u8(graph, 1, config.as_bytes())
                }
                None => {
                    let err_msg = format!(
                        "The model `{}` does not exist in the chat graphs.",
                        &model_name
                    );

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        None => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. Reason: {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    // update metadata
                    set_tensor_data_u8(graph, 1, config.as_bytes())
                }
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    }
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
}
impl ChatStream {
    fn new(model: Option<String>, id: String, include_usage: bool) -> Self {
        let stream_state = if include_usage {
            StreamState::Usage
        } else {
            StreamState::Done
        };

        ChatStream {
            id,
            model,
            include_usage,
            context_full_state: ContextFullState::Message,
            prompt_too_long_state: PromptTooLongState::Message,
            stream_state,
        }
    }
}
impl Drop for ChatStream {
    fn drop(&mut self) {
        match &self.model {
            Some(model_name) => {
                match CHAT_GRAPHS.get() {
                    Some(chat_graphs) => match chat_graphs.lock() {
                        Ok(mut chat_graphs) => match chat_graphs.get_mut(model_name) {
                            Some(graph) => {
                                if let Err(e) = graph.finish_single() {
                                    let err_msg =
                                        format!("Failed to clean up the context. Reason: {}", e);

                                    #[cfg(feature = "logging")]
                                    error!(target: "llama_core", "{}", &err_msg);

                                    #[cfg(not(feature = "logging"))]
                                    println!(
                                        "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                        &err_msg
                                    );
                                }
                            }
                            None => {
                                let err_msg = format!(
                                    "The model `{}` does not exist in the chat graphs.",
                                    &model_name
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "llama_core", "{}", &err_msg);

                                #[cfg(not(feature = "logging"))]
                                println!(
                                    "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                    &err_msg
                                );
                            }
                        },
                        Err(e) => {
                            let err_msg =
                                format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "llama_core", "{}", &err_msg);

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
                        error!(target: "llama_core", "{}", &err_msg);

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
                                    let err_msg =
                                        format!("Failed to clean up the context. Reason: {}", e);

                                    #[cfg(feature = "logging")]
                                    error!(target: "llama_core", "{}", &err_msg);

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
                                error!(target: "llama_core", "{}", err_msg);

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
                            error!(target: "llama_core", "{}", &err_msg);

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
                        error!(target: "llama_core", "{}", &err_msg);

                        #[cfg(not(feature = "logging"))]
                        println!(
                            "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                            &err_msg
                        );
                    }
                };
            }
        }
    }
}
impl futures::Stream for ChatStream {
    type Item = Result<String, LlamaCoreError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let x = compute_stream(
            this.model.clone(),
            this.id.clone(),
            this.include_usage,
            &mut this.prompt_too_long_state,
            &mut this.context_full_state,
            &mut this.stream_state,
        );

        match x {
            Ok(x) => {
                if x != "[GGML] End of sequence" && !x.is_empty() {
                    Poll::Ready(Some(Ok(x)))
                } else {
                    // stopped
                    Poll::Ready(None)
                }
            }
            Err(e) => Poll::Ready(Some(Err(e))),
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
    if *prompt_too_long_state == PromptTooLongState::EndOfSequence
        || *context_full_state == ContextFullState::EndOfSequence
        || *stream_state == StreamState::EndOfSequence
    {
        return Ok("[GGML] End of sequence".to_string());
    }

    // get graph
    match &model_name {
        Some(model_name) => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match chat_graphs.get_mut(model_name) {
                Some(graph) => {
                    // compute
                    match graph.compute_single() {
                        Ok(_) => {
                            // Retrieve the output
                            let output_buffer = get_output_buffer_single(graph, OUTPUT_TENSOR)?;

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
                                            error!(target: "llama_core", "{}", &err_msg);


                                            LlamaCoreError::Operation(err_msg)
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
                                            // TODO This is a temp check. In case, infinite cached encodings happen.
                                            if cached_encodings.len() > 3 {
                                                let err_msg = "The length of the invalid utf8 bytes exceed 3.";

                                                #[cfg(feature = "logging")]
                                                error!(target: "llama_core", "{}", &err_msg);

                                                return Err(LlamaCoreError::Operation(
                                                    err_msg.into(),
                                                ));
                                            }

                                            String::new()
                                        }
                                    }
                                }
                            };

                            let created = SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map_err(|e| {
                                let err_msg =
                                    format!("Failed to get the current time. Reason: {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "llama_core", "{}", &err_msg);

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
                                    let err_msg = format!(
                                        "Failed to serialize chat completion chunk. Reason: {}",
                                        e
                                    );

                                    #[cfg(feature = "logging")]
                                    error!(target: "llama_core", "{}", &err_msg);

                                    LlamaCoreError::Operation(err_msg)
                                })?;

                            Ok(format!("data: {}\n\n", chunk_str))
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::EndOfSequence,
                        )) => {
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
                                    info!(target: "llama_core", "token_info: {} prompt tokens, {} completion tokens", token_info.prompt_tokens, token_info.completion_tokens);

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "llama_core", "{}", &err_msg);

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
                                        error!(target: "llama_core", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                StreamState::Done => {
                                    *stream_state = StreamState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                StreamState::EndOfSequence => {
                                    // clear context
                                    if let Err(e) = graph.finish_single() {
                                        let err_msg = format!(
                                            "Failed to clean up the context. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "llama_core", "{}", &err_msg);

                                        return Err(LlamaCoreError::Backend(
                                            BackendError::FinishSingle(err_msg),
                                        ));
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
                                            error!(target: "llama_core", "{}", &err_msg);

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
                                                role: Some(ChatCompletionRole::Assistant),
                                                content: Some(
                                                    "<|WASMEDGE-GGML-CONTEXT-FULL|>".to_string(),
                                                ),
                                                function_call: None,
                                                tool_calls: None,
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
                                        error!(target: "llama_core", "{}", &err_msg);

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
                                            error!(target: "llama_core", "{}", &err_msg);

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
                                        error!(target: "llama_core", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                ContextFullState::Done => {
                                    *context_full_state = ContextFullState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                ContextFullState::EndOfSequence => {
                                    // clear context
                                    if let Err(e) = graph.finish_single() {
                                        let err_msg = format!(
                                            "Failed to clean up the context. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "llama_core", "{}", &err_msg);

                                        return Err(LlamaCoreError::Backend(
                                            BackendError::FinishSingle(err_msg),
                                        ));
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
                                            error!(target: "llama_core", "{}", &err_msg);

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
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "llama_core", "{}", &err_msg);

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
                                            error!(target: "llama_core", "{}", &err_msg);

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
                                        error!(target: "llama_core", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                PromptTooLongState::Done => {
                                    *prompt_too_long_state = PromptTooLongState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                PromptTooLongState::EndOfSequence => {
                                    // clear context
                                    if let Err(e) = graph.finish_single() {
                                        let err_msg = format!(
                                            "Failed to clean up the context. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "llama_core", "{}", &err_msg);

                                        return Err(LlamaCoreError::Backend(
                                            BackendError::FinishSingle(err_msg),
                                        ));
                                    }

                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(e) => {
                            // clear context
                            if let Err(e) = graph.finish_single() {
                                let err_msg =
                                    format!("Failed to clean up the context. Reason: {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "llama_core", "{}", &err_msg);

                                return Err(LlamaCoreError::Backend(BackendError::FinishSingle(
                                    err_msg,
                                )));
                            }

                            let err_msg =
                                format!("Failed to compute the chat completion. Reason: {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "llama_core", "{}", &err_msg);

                            Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                                err_msg,
                            )))
                        }
                    }
                }
                None => {
                    let err_msg = format!(
                        "The model `{}` does not exist in the chat graphs.",
                        &model_name
                    );

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        None => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "llama_core", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    // compute
                    match graph.compute_single() {
                        Ok(_) => {
                            // Retrieve the output
                            let output_buffer = get_output_buffer_single(graph, OUTPUT_TENSOR)?;
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
                                            error!(target: "llama_core", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    cached_encodings.extend_from_slice(&output_buffer[..]);

                                    match String::from_utf8(cached_encodings.to_vec()) {
                                        Ok(token) => {
                                            // clear encodings
                                            cached_encodings.clear();

                                            token
                                        }
                                        Err(_) => {
                                            // TODO This is a temp check. In case, infinite cached encodings happen.
                                            if cached_encodings.len() > 3 {
                                                let err_msg = "The length of the invalid utf8 bytes exceed 3.";

                                                #[cfg(feature = "logging")]
                                                error!(target: "llama_core", "{}", &err_msg);

                                                return Err(LlamaCoreError::Operation(
                                                    err_msg.into(),
                                                ));
                                            }

                                            String::new()
                                        }
                                    }
                                }
                            };

                            let created = SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map_err(|e| {
                                let err_msg =
                                    format!("Failed to get the current time. Reason: {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "llama_core", "{}", &err_msg);

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
                                    let err_msg = format!(
                                        "Failed to serialize chat completion chunk. Reason: {}",
                                        e
                                    );

                                    #[cfg(feature = "logging")]
                                    error!(target: "llama_core", "{}", &err_msg);

                                    LlamaCoreError::Operation(err_msg)
                                })?;

                            Ok(format!("data: {}\n\n", chunk_str))
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::EndOfSequence,
                        )) => {
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
                                    info!(target: "llama_core", "token_info: {} prompt tokens, {} completion tokens", token_info.prompt_tokens, token_info.completion_tokens);

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "llama_core", "{}", &err_msg);

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
                                        error!(target: "llama_core", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                StreamState::Done => {
                                    *stream_state = StreamState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                StreamState::EndOfSequence => {
                                    // clear context
                                    if let Err(e) = graph.finish_single() {
                                        let err_msg = format!(
                                            "Failed to clean up the context. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "llama_core", "{}", &err_msg);

                                        return Err(LlamaCoreError::Backend(
                                            BackendError::FinishSingle(err_msg),
                                        ));
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
                                            error!(target: "llama_core", "{}", &err_msg);

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
                                                role: Some(ChatCompletionRole::Assistant),
                                                content: Some(
                                                    "<|WASMEDGE-GGML-CONTEXT-FULL|>".to_string(),
                                                ),
                                                function_call: None,
                                                tool_calls: None,
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
                                        error!(target: "llama_core", "{}", &err_msg);

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
                                            error!(target: "llama_core", "{}", &err_msg);

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
                                        error!(target: "llama_core", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                ContextFullState::Done => {
                                    *context_full_state = ContextFullState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                ContextFullState::EndOfSequence => {
                                    // clear context
                                    if let Err(e) = graph.finish_single() {
                                        let err_msg = format!(
                                            "Failed to clean up the context. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "llama_core", "{}", &err_msg);

                                        return Err(LlamaCoreError::Backend(
                                            BackendError::FinishSingle(err_msg),
                                        ));
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
                                            error!(target: "llama_core", "{}", &err_msg);

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
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "llama_core", "{}", &err_msg);

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
                                            error!(target: "llama_core", "{}", &err_msg);

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
                                        error!(target: "llama_core", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                PromptTooLongState::Done => {
                                    *prompt_too_long_state = PromptTooLongState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                PromptTooLongState::EndOfSequence => {
                                    // clear context
                                    if let Err(e) = graph.finish_single() {
                                        let err_msg = format!(
                                            "Failed to clean up the context. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "llama_core", "{}", &err_msg);

                                        return Err(LlamaCoreError::Backend(
                                            BackendError::FinishSingle(err_msg),
                                        ));
                                    }

                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(e) => {
                            // clear context
                            if let Err(e) = graph.finish_single() {
                                let err_msg =
                                    format!("Failed to clean up the context. Reason: {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "llama_core", "{}", &err_msg);

                                return Err(LlamaCoreError::Backend(BackendError::FinishSingle(
                                    err_msg,
                                )));
                            }

                            let err_msg =
                                format!("Failed to compute the chat completion. Reason: {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "llama_core", "{}", &err_msg);

                            Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                                err_msg,
                            )))
                        }
                    }
                }
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    }
}
