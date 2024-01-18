use crate::{
    error, print_log_begin_separator, print_log_end_separator, Graph, ModelInfo, CTX_SIZE, GRAPH,
    MAX_BUFFER_SIZE, METADATA,
};
use chat_prompts::{
    chat::{
        belle::BelleLlama2ChatPrompt,
        llama::{CodeLlamaInstructPrompt, Llama2ChatPrompt},
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
        ChatCompletionRequest, ChatCompletionRole,
    },
    common::{FinishReason, Usage},
    completions::{CompletionChoice, CompletionObject, CompletionRequest},
    models::{ListModelsResponse, Model},
};
use futures::{future, stream};
use futures_util::TryStreamExt;
use hyper::{body::to_bytes, Body, Request, Response};
use serde_json::Value;
use std::time::SystemTime;

/// Lists models available
pub(crate) async fn models_handler(
    model_info: ModelInfo,
    template_ty: PromptTemplateType,
    created: u64,
) -> Result<Response<Body>, hyper::Error> {
    let model = Model {
        id: format!(
            "{name}:{template}",
            name = model_info.name,
            template = template_ty.to_string()
        ),
        created: created.clone(),
        object: String::from("model"),
        owned_by: String::from("Not specified"),
    };

    let list_models_response = ListModelsResponse {
        object: String::from("list"),
        data: vec![model],
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .body(Body::from(
            serde_json::to_string(&list_models_response).unwrap(),
        ));
    match result {
        Ok(response) => Ok(response),
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

pub(crate) async fn _embeddings_handler() -> Result<Response<Body>, hyper::Error> {
    println!("llama_embeddings_handler not implemented");
    error::not_implemented()
}

pub(crate) async fn completions_handler(
    mut req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    println!("[COMPLETION] New completion begins ...");

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let completion_request: CompletionRequest = serde_json::from_slice(&body_bytes).unwrap();

    let prompt = completion_request.prompt.join(" ");

    // ! todo: a temp solution of computing the number of tokens in prompt
    let prompt_tokens = prompt.split_whitespace().count() as u32;

    let buffer = match infer(prompt.trim()).await {
        Ok(buffer) => buffer,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    // convert inference result to string
    let model_answer = String::from_utf8(buffer.clone()).unwrap();
    let answer = model_answer.trim();

    // ! todo: a temp solution of computing the number of tokens in answer
    let completion_tokens = answer.split_whitespace().count() as u32;

    println!("[COMPLETION] Bot answer: {}", answer);

    println!("[COMPLETION] New completion ends.");

    let completion_object = CompletionObject {
        id: uuid::Uuid::new_v4().to_string(),
        object: String::from("text_completion"),
        created: SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: completion_request.model.clone().unwrap_or_default(),
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
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .body(Body::from(
            serde_json::to_string(&completion_object).unwrap(),
        ));
    match result {
        Ok(response) => Ok(response),
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

/// Processes a chat-completion request and returns a chat-completion response with the answer from the model.
pub(crate) async fn chat_completions_handler(
    mut req: Request<Body>,
    template_ty: PromptTemplateType,
    log_prompts: bool,
) -> Result<Response<Body>, hyper::Error> {
    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .body(Body::empty());

        match result {
            Ok(response) => return Ok(response),
            Err(e) => {
                return error::internal_server_error(e.to_string());
            }
        }
    }

    fn create_prompt_template(template_ty: PromptTemplateType) -> ChatPrompt {
        match template_ty {
            PromptTemplateType::Llama2Chat => {
                ChatPrompt::Llama2ChatPrompt(Llama2ChatPrompt::default())
            }
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
            PromptTemplateType::BelleLlama2Chat => {
                ChatPrompt::BelleLlama2ChatPrompt(BelleLlama2ChatPrompt::default())
            }
            PromptTemplateType::VicunaChat => {
                ChatPrompt::VicunaChatPrompt(chat_prompts::chat::vicuna::VicunaChatPrompt::default())
            }
            PromptTemplateType::Vicuna11Chat => ChatPrompt::Vicuna11ChatPrompt(
                chat_prompts::chat::vicuna::Vicuna11ChatPrompt::default(),
            ),
            PromptTemplateType::ChatML => {
                ChatPrompt::ChatMLPrompt(chat_prompts::chat::chatml::ChatMLPrompt::default())
            }
            PromptTemplateType::Baichuan2 => ChatPrompt::Baichuan2ChatPrompt(
                chat_prompts::chat::baichuan::Baichuan2ChatPrompt::default(),
            ),
            PromptTemplateType::WizardCoder => ChatPrompt::WizardCoderPrompt(
                chat_prompts::chat::wizard::WizardCoderPrompt::default(),
            ),
            PromptTemplateType::Zephyr => {
                ChatPrompt::ZephyrChatPrompt(chat_prompts::chat::zephyr::ZephyrChatPrompt::default())
            }
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
        }
    }
    let template = create_prompt_template(template_ty);

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut chat_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chat_request) => chat_request,
        Err(e) => {
            return error::bad_request(format!(
                "Fail to parse chat completion request: {msg}",
                msg = e.to_string()
            ));
        }
    };

    // build prompt
    let prompt = match build_prompt(&template, &mut chat_request) {
        Ok(prompt) => prompt,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    if log_prompts {
        print_log_begin_separator("PROMPT", Some("*"), None);
        println!("\n{}", &prompt,);
        print_log_end_separator(Some("*"), None);
    }

    // update metadata
    if let Err(msg) = update_metadata(&chat_request) {
        return error::internal_server_error(msg);
    }

    let mut graph = crate::GRAPH.get().unwrap().lock().unwrap();

    // set input
    let tensor_data = prompt.as_bytes().to_vec();
    if graph
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .is_err()
    {
        return error::internal_server_error(String::from("Fail to set input tensor"));
    };

    let result = match chat_request.stream {
        Some(_) => {
            let model = chat_request.model.clone().unwrap_or_default();
            let mut one_more_run_then_stop = true;
            let stream = stream::repeat_with(move || {
                let mut graph = crate::GRAPH.get().unwrap().lock().unwrap();

                // compute
                match graph.compute_single() {
                    Ok(_) => {
                        // Retrieve the output.
                        let mut output_buffer = vec![0u8; *MAX_BUFFER_SIZE.get().unwrap()];
                        let mut output_size = match graph.get_output_single(0, &mut output_buffer) {
                            Ok(size) => size,
                            Err(e) => {
                                return Err(format!(
                                    "Fail to get output tensor: {msg}",
                                    msg = e.to_string()
                                ));
                            }
                        };
                        output_size = std::cmp::min(*MAX_BUFFER_SIZE.get().unwrap(), output_size);

                        let output =
                            String::from_utf8_lossy(&output_buffer[..output_size]).to_string();

                        let chat_completion_chunk = ChatCompletionChunk {
                            id: "chatcmpl-123".to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
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

                        Ok(serde_json::to_string(&chat_completion_chunk).unwrap())
                    }
                    Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::EndOfSequence)) => {
                        match one_more_run_then_stop {
                            true => {
                                let chat_completion_chunk = ChatCompletionChunk {
                                    id: "chatcmpl-123".to_string(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap()
                                        .as_secs(),
                                    model: model.clone(),
                                    system_fingerprint: "fp_44709d6fcb".to_string(),
                                    choices: vec![ChatCompletionChunkChoice {
                                        index: 0,
                                        delta: ChatCompletionChunkChoiceDelta {
                                            role: Some(ChatCompletionRole::Assistant),
                                            content: Some("<|WASMEDGE-GGML-EOS|>".to_string()),
                                            function_call: None,
                                            tool_calls: None,
                                        },
                                        logprobs: None,
                                        finish_reason: Some(FinishReason::stop),
                                    }],
                                };
                                one_more_run_then_stop = false;
                                Ok(serde_json::to_string(&chat_completion_chunk).unwrap())
                            }
                            false => Ok("[GGML] End of sequence".to_string()),
                        }
                    }
                    Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::ContextFull)) => {
                        println!(
                            "\n\n[WARNING] The message is cut off as the max context size is reached. You can try to ask the same question again, or increase the context size via the `--ctx-size` command option.\n"
                        );

                        let chat_completion_chunk = ChatCompletionChunk {
                            id: "chatcmpl-123".to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
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

                        if let Err(e) = graph.finish_single() {
                            println!("Error: {:?}", &e);
                            return Err(e.to_string());
                        }

                        Ok(serde_json::to_string(&chat_completion_chunk).unwrap())
                    }
                    Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::PromptTooLong)) => {
                        println!("\n\n[WARNING] The prompt is too long.");

                        let chat_completion_chunk = ChatCompletionChunk {
                            id: "chatcmpl-123".to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
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

                        if let Err(e) = graph.finish_single() {
                            println!("Error: {:?}", &e);
                            return Err(e.to_string());
                        }

                        Ok(serde_json::to_string(&chat_completion_chunk).unwrap())
                    }
                    Err(e) => {
                        println!("Error: {:?}", &e);
                        return Err(e.to_string());
                    }
                }
            });

            // create hyer stream
            let stream =
                stream.try_take_while(|x| future::ready(Ok(x != "[GGML] End of sequence")));

            Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .body(Body::wrap_stream(stream))
        }
        None => {
            match graph.compute() {
                Ok(_) => {
                    // Retrieve the output.
                    let mut output_buffer = vec![0u8; *MAX_BUFFER_SIZE.get().unwrap()];
                    let mut output_size = match graph.get_output(0, &mut output_buffer) {
                        Ok(size) => size,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Fail to get output tensor: {msg}",
                                msg = e.to_string()
                            ));
                        }
                    };
                    output_size = std::cmp::min(*MAX_BUFFER_SIZE.get().unwrap(), output_size);
                    // convert inference result to string
                    let output = std::str::from_utf8(&output_buffer[..output_size]).unwrap();

                    // post-process
                    let message = post_process(&output, template_ty);

                    // retrieve the number of prompt and completion tokens
                    let token_info = get_token_info(&graph);
                    if log_prompts {
                        print_log_begin_separator("PROMPT (Tokens)", Some("*"), None);
                        println!(
                            "\nprompt tokens: {}, completion_tokens: {}",
                            token_info.prompt_tokens, token_info.completion_tokens
                        );
                        print_log_end_separator(Some("*"), None);
                    }

                    // create ChatCompletionResponse
                    let chat_completion_obejct = ChatCompletionObject {
                        id: uuid::Uuid::new_v4().to_string(),
                        object: String::from("chat.completion"),
                        created: SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
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
                            prompt_tokens: token_info.prompt_tokens as u32,
                            completion_tokens: token_info.completion_tokens as u32,
                            total_tokens: token_info.prompt_tokens as u32
                                + token_info.completion_tokens as u32,
                        },
                    };

                    // return response
                    Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .body(Body::from(
                            serde_json::to_string(&chat_completion_obejct).unwrap(),
                        ))
                }
                Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::ContextFull)) => {
                    println!(
                        "\n\n[WARNING] The message is cut off as the max context size is reached. Try to ask the same question again, or increase the context size via the `--ctx-size` command option.\n"
                    );

                    // Retrieve the output.
                    let max_buffer_size = *MAX_BUFFER_SIZE.get().unwrap();
                    let mut output_buffer = vec![0u8; max_buffer_size];
                    let mut output_size = match graph.get_output(0, &mut output_buffer) {
                        Ok(size) => size,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Fail to get output tensor: {msg}",
                                msg = e.to_string()
                            ));
                        }
                    };
                    output_size = std::cmp::min(max_buffer_size, output_size);
                    // convert inference result to string
                    let output = std::str::from_utf8(&output_buffer[..output_size]).unwrap();

                    // post-process
                    let message = post_process(&output, template_ty);

                    // retrieve the number of prompt and completion tokens
                    let token_info = get_token_info(&graph);
                    if log_prompts {
                        print_log_begin_separator("PROMPT (Tokens)", Some("*"), None);
                        println!(
                            "\nprompt tokens: {}, completion_tokens: {}",
                            token_info.prompt_tokens, token_info.completion_tokens
                        );
                        print_log_end_separator(Some("*"), None);
                    }

                    // create ChatCompletionResponse
                    let chat_completion_obejct = ChatCompletionObject {
                        id: uuid::Uuid::new_v4().to_string(),
                        object: String::from("chat.completion"),
                        created: SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
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
                            prompt_tokens: token_info.prompt_tokens as u32,
                            completion_tokens: token_info.completion_tokens as u32,
                            total_tokens: token_info.prompt_tokens as u32
                                + token_info.completion_tokens as u32,
                        },
                    };

                    // return response
                    Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .body(Body::from(
                            serde_json::to_string(&chat_completion_obejct).unwrap(),
                        ))
                }
                Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::PromptTooLong)) => {
                    println!("\n\n[WARNING] The prompt is too long.\n");

                    // Retrieve the output.
                    let mut output_buffer = vec![0u8; *MAX_BUFFER_SIZE.get().unwrap()];
                    let mut output_size = match graph.get_output(0, &mut output_buffer) {
                        Ok(size) => size,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Fail to get output tensor: {msg}",
                                msg = e.to_string()
                            ));
                        }
                    };
                    output_size = std::cmp::min(*MAX_BUFFER_SIZE.get().unwrap(), output_size);
                    // convert inference result to string
                    let output = std::str::from_utf8(&output_buffer[..output_size]).unwrap();

                    // post-process
                    let message = post_process(&output, template_ty);

                    // retrieve the number of prompt and completion tokens
                    let token_info = get_token_info(&graph);
                    if log_prompts {
                        print_log_begin_separator("PROMPT (Tokens)", Some("*"), None);
                        println!(
                            "\nprompt tokens: {}, completion_tokens: {}",
                            token_info.prompt_tokens, token_info.completion_tokens
                        );
                        print_log_end_separator(Some("*"), None);
                    }

                    // create ChatCompletionResponse
                    let chat_completion_obejct = ChatCompletionObject {
                        id: uuid::Uuid::new_v4().to_string(),
                        object: String::from("chat.completion"),
                        created: SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
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
                            prompt_tokens: token_info.prompt_tokens as u32,
                            completion_tokens: token_info.completion_tokens as u32,
                            total_tokens: token_info.completion_tokens as u32
                                + token_info.completion_tokens as u32,
                        },
                    };

                    // return response
                    Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .body(Body::from(
                            serde_json::to_string(&chat_completion_obejct).unwrap(),
                        ))
                }
                Err(e) => {
                    println!("Error: {:?}", &e);
                    return error::internal_server_error(e.to_string());
                }
            }
        }
    };

    match result {
        Ok(response) => Ok(response),
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

/// Runs inference on the model with the given name and returns the output.
pub(crate) async fn infer(prompt: impl AsRef<str>) -> std::result::Result<Vec<u8>, String> {
    let mut graph = crate::GRAPH.get().unwrap().lock().unwrap();

    // set input
    let tensor_data = prompt.as_ref().as_bytes().to_vec();
    if graph
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .is_err()
    {
        return Err(String::from("Fail to set input tensor"));
    };

    // execute the inference
    if graph.compute().is_err() {
        return Err(String::from("Fail to execute model inference"));
    }

    // Retrieve the output.
    let mut output_buffer = vec![0u8; *MAX_BUFFER_SIZE.get().unwrap()];
    let mut output_size = match graph.get_output(0, &mut output_buffer) {
        Ok(size) => size,
        Err(e) => {
            return Err(format!(
                "Fail to get output tensor: {msg}",
                msg = e.to_string()
            ))
        }
    };
    output_size = std::cmp::min(*MAX_BUFFER_SIZE.get().unwrap(), output_size);

    Ok(output_buffer[..output_size].to_vec())
}

fn update_metadata(chat_request: &ChatCompletionRequest) -> Result<(), String> {
    let mut should_update = false;
    let mut metadata = METADATA.get().unwrap().clone();

    // check if necessary to update n_predict with max_tokens
    if let Some(max_tokens) = chat_request.max_tokens {
        let max_tokens = max_tokens as u64;
        if metadata.n_predict > max_tokens {
            // update n_predict
            metadata.n_predict = max_tokens;

            if !should_update {
                should_update = true;
            }
        }
    }

    // check if necessary to update temperature
    if let Some(temp) = chat_request.temperature {
        if metadata.temp != temp {
            // update temperature
            metadata.temp = temp;

            if !should_update {
                should_update = true;
            }
        }
    }

    // check if necessary to update repetition_penalty
    if let Some(repeat_penalty) = chat_request.frequency_penalty {
        if metadata.repeat_penalty != repeat_penalty {
            // update repetition_penalty
            metadata.repeat_penalty = repeat_penalty;

            if !should_update {
                should_update = true;
            }
        }
    }

    if should_update {
        let mut graph = GRAPH.get().unwrap().lock().unwrap();

        // update metadata
        let config = serde_json::to_string(&metadata).unwrap();
        if graph
            .set_input(1, wasi_nn::TensorType::U8, &[1], config.as_bytes())
            .is_err()
        {
            return Err(String::from("Fail to update metadata"));
        }
    }

    Ok(())
}

fn post_process(output: impl AsRef<str>, template_ty: PromptTemplateType) -> String {
    if template_ty == PromptTemplateType::Baichuan2 {
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
                .to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if template_ty == PromptTemplateType::BelleLlama2Chat {
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
    }
}

fn build_prompt(
    template: &ChatPrompt,
    chat_request: &mut ChatCompletionRequest,
    // graph: &mut Graph,
    // max_prompt_tokens: u64,
) -> Result<String, String> {
    let mut graph = GRAPH.get().unwrap().lock().unwrap();
    let max_prompt_tokens = *CTX_SIZE.get().unwrap() as u64 * 4 / 5;
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
        let max_input_size = *MAX_BUFFER_SIZE.get().unwrap();
        let mut input_buffer = vec![0u8; max_input_size];
        let mut input_size = graph.get_output(1, &mut input_buffer).unwrap();
        input_size = std::cmp::min(max_input_size, input_size);
        let token_info: Value = serde_json::from_slice(&input_buffer[..input_size]).unwrap();
        let prompt_tokens = token_info["input_tokens"].as_u64().unwrap();

        match prompt_tokens > max_prompt_tokens {
            true => {
                match chat_request.messages[0].role {
                    ChatCompletionRole::System => {
                        if chat_request.messages.len() >= 4 {
                            if chat_request.messages[1].role == ChatCompletionRole::User {
                                chat_request.messages.remove(1);
                            }
                            if chat_request.messages[1].role == ChatCompletionRole::Assistant {
                                chat_request.messages.remove(1);
                            }
                        } else if chat_request.messages.len() == 3
                            && chat_request.messages[1].role == ChatCompletionRole::User
                        {
                            chat_request.messages.remove(1);
                        } else {
                            return Ok(prompt);
                        }
                    }
                    ChatCompletionRole::User => {
                        if chat_request.messages.len() >= 3 {
                            if chat_request.messages[0].role == ChatCompletionRole::User {
                                chat_request.messages.remove(0);
                            }
                            if chat_request.messages[0].role == ChatCompletionRole::Assistant {
                                chat_request.messages.remove(0);
                            }
                        } else if chat_request.messages.len() == 2
                            && chat_request.messages[0].role == ChatCompletionRole::User
                        {
                            chat_request.messages.remove(0);
                        } else {
                            return Ok(prompt);
                        }
                    }
                    _ => panic!("Found a unsupported chat message role!"),
                }
                continue;
            }
            false => return Ok(prompt),
        }
    }
}

fn get_token_info(graph: &Graph) -> TokenInfo {
    let max_output_size = *MAX_BUFFER_SIZE.get().unwrap();
    let mut output_buffer = vec![0u8; max_output_size];
    let mut output_size = graph.get_output(1, &mut output_buffer).unwrap();
    output_size = std::cmp::min(max_output_size, output_size);
    let token_info: Value = serde_json::from_slice(&output_buffer[..output_size]).unwrap();
    TokenInfo {
        prompt_tokens: token_info["input_tokens"].as_u64().unwrap(),
        completion_tokens: token_info["output_tokens"].as_u64().unwrap(),
    }
}

#[derive(Debug)]

struct TokenInfo {
    prompt_tokens: u64,
    completion_tokens: u64,
}
