use crate::{
    error, print_log_begin_separator, print_log_end_separator, Graph, ModelInfo, CTX_SIZE, GRAPH,
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
use std::{sync::Mutex, time::SystemTime};

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

    // serialize response
    let s = match serde_json::to_string(&list_models_response) {
        Ok(s) => s,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .body(Body::from(s));
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
    let completion_request: CompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(completion_request) => completion_request,
        Err(e) => {
            return error::bad_request(format!(
                "Failed to deserialize completion request. {msg}",
                msg = e.to_string()
            ));
        }
    };

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
    let model_answer = match String::from_utf8(buffer.clone()) {
        Ok(model_answer) => model_answer,
        Err(e) => {
            return error::internal_server_error(format!(
                "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                e.to_string()
            ));
        }
    };
    let answer = model_answer.trim();

    // ! todo: a temp solution of computing the number of tokens in answer
    let completion_tokens = answer.split_whitespace().count() as u32;

    println!("[COMPLETION] Bot answer: {}", answer);

    println!("[COMPLETION] New completion ends.");

    let created = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(created) => created.as_secs(),
        Err(e) => {
            return error::internal_server_error(format!(
                "Failed to get the current time. {}",
                e.to_string()
            ));
        }
    };

    let completion_object = CompletionObject {
        id: uuid::Uuid::new_v4().to_string(),
        object: String::from("text_completion"),
        created,
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

    // serialize completion object
    let s = match serde_json::to_string(&completion_object) {
        Ok(s) => s,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .body(Body::from(s));
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
            PromptTemplateType::CodeLlamaSuper => {
                ChatPrompt::CodeLlamaSuperInstructPrompt(CodeLlamaSuperInstructPrompt::default())
            }
            PromptTemplateType::HumanAssistant => {
                ChatPrompt::HumanAssistantChatPrompt(HumanAssistantChatPrompt::default())
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
            PromptTemplateType::Phi2Instruct => ChatPrompt::Phi2InstructPrompt(
                chat_prompts::chat::phi::Phi2InstructPrompt::default(),
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
    let (prompt, avaible_completion_tokens) = match build_prompt(&template, &mut chat_request) {
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
    if let Err(msg) = update_metadata(&chat_request, avaible_completion_tokens) {
        return error::internal_server_error(msg);
    }

    let graph = match GRAPH.get() {
        Some(graph) => graph,
        None => {
            return error::internal_server_error(String::from(
                "Fail to get the underlying value of `GRAPH`.",
            ));
        }
    };
    let mut graph = match graph.lock() {
        Ok(graph) => graph,
        Err(e) => {
            return error::internal_server_error(format!(
                "Fail to acquire the lock of `GRAPH`. {}",
                e.to_string()
            ));
        }
    };

    // set input
    let tensor_data = prompt.as_bytes().to_vec();
    if graph
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .is_err()
    {
        return error::internal_server_error(String::from("Fail to set input tensor"));
    };

    let result = match chat_request.stream {
        Some(true) => {
            let model = chat_request.model.clone().unwrap_or_default();

            let stop = {
                let metadata = match METADATA.get() {
                    Some(metadata) => metadata.clone(),
                    None => {
                        return error::internal_server_error(String::from(
                            "Fail to get the underlying value of `METADATA`.",
                        ));
                    }
                };
                metadata.reverse_prompt
            };
            let ref_stop = std::sync::Arc::new(stop);

            // let mutex = UTF8_ENCODINGS.get_or_init(|| Mutex::new(Vec::new()));

            let mut one_more_run_then_stop = true;
            let stream = stream::repeat_with(move || {
                let reverse_prompt = ref_stop.clone();
                // let invalid_utf8_vec = ref_invalid_utf8_vec.clone();

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

                // compute
                match graph.compute_single() {
                    Ok(_) => {
                        match one_more_run_then_stop {
                            true => {
                                // Retrieve the output.
                                let max_buffer_size = match MAX_BUFFER_SIZE.get() {
                                    Some(max_buffer_size) => max_buffer_size,
                                    None => {
                                        return Err(String::from(
                                            "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
                                        ));
                                    }
                                };
                                let mut output_buffer = vec![0u8; *max_buffer_size];
                                let mut output_size =
                                    match graph.get_output_single(0, &mut output_buffer) {
                                        Ok(size) => size,
                                        Err(e) => {
                                            return Err(format!(
                                                "Fail to get output tensor: {msg}",
                                                msg = e.to_string()
                                            ));
                                        }
                                    };
                                output_size = std::cmp::min(*max_buffer_size, output_size);

                                // decode the output buffer to a utf8 string
                                let output = match String::from_utf8(
                                    output_buffer[..output_size].to_vec(),
                                ) {
                                    Ok(token) => token,
                                    Err(_) => {
                                        let mutex =
                                            UTF8_ENCODINGS.get_or_init(|| Mutex::new(Vec::new()));

                                        let mut encodings = match mutex.lock() {
                                            Ok(encodings) => encodings,
                                            Err(e) => {
                                                return Err(format!(
                                                    "Fail to acquire the lock of `UTF8_ENCODINGS`. {}",
                                                    e.to_string()
                                                ));
                                            }
                                        };
                                        encodings.extend_from_slice(&output_buffer[..output_size]);

                                        if encodings.len() > 3 {
                                            return Err(String::from(
                                                "The length of the invalid utf8 bytes exceed 3.",
                                            ));
                                        }

                                        if encodings.len() == 3 {
                                            let token = match String::from_utf8(encodings.to_vec())
                                            {
                                                Ok(token) => token,
                                                Err(e) => {
                                                    return Err(format!(
                                                        "Failed to decode the invalid utf8 bytes to a utf8 string. {}",
                                                        e.to_string()
                                                    ));
                                                }
                                            };

                                            // clear encodings
                                            encodings.clear();

                                            token
                                        } else {
                                            String::new()
                                        }
                                    }
                                };

                                if let Some(stop) = &*reverse_prompt.clone() {
                                    if output == *stop {
                                        let created = match SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                        {
                                            Ok(created) => created.as_secs(),
                                            Err(e) => {
                                                return Err(format!(
                                                    "Failed to get the current time. {}",
                                                    e.to_string()
                                                ));
                                            }
                                        };

                                        let chat_completion_chunk = ChatCompletionChunk {
                                            id: "chatcmpl-123".to_string(),
                                            object: "chat.completion.chunk".to_string(),
                                            created,
                                            model: model.clone(),
                                            system_fingerprint: "fp_44709d6fcb".to_string(),
                                            choices: vec![ChatCompletionChunkChoice {
                                                index: 0,
                                                delta: ChatCompletionChunkChoiceDelta {
                                                    role: Some(ChatCompletionRole::Assistant),
                                                    content: Some(
                                                        "<|WASMEDGE-GGML-EOS|>".to_string(),
                                                    ),
                                                    function_call: None,
                                                    tool_calls: None,
                                                },
                                                logprobs: None,
                                                finish_reason: Some(FinishReason::stop),
                                            }],
                                        };

                                        if let Err(e) = graph.finish_single() {
                                            println!("Error: {:?}", &e);
                                            return Err(e.to_string());
                                        }

                                        one_more_run_then_stop = false;

                                        // serialize chat completion chunk
                                        let chunk =
                                            match serde_json::to_string(&chat_completion_chunk) {
                                                Ok(chunk) => chunk,
                                                Err(e) => {
                                                    return Err(format!(
                                            "Fail to serialize chat completion chunk. {}",
                                            e.to_string()
                                        ));
                                                }
                                            };

                                        return Ok(chunk);
                                    }
                                }

                                let created =
                                    match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                                        Ok(created) => created.as_secs(),
                                        Err(e) => {
                                            return Err(format!(
                                                "Failed to get the current time. {}",
                                                e.to_string()
                                            ));
                                        }
                                    };
                                let chat_completion_chunk = ChatCompletionChunk {
                                    id: "chatcmpl-123".to_string(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
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
                                let chunk = match serde_json::to_string(&chat_completion_chunk) {
                                    Ok(chunk) => chunk,
                                    Err(e) => {
                                        return Err(format!(
                                            "Fail to serialize chat completion chunk. {}",
                                            e.to_string()
                                        ));
                                    }
                                };

                                Ok(chunk)
                            }
                            false => {
                                return Ok("[GGML] End of sequence".to_string());
                            }
                        }
                    }
                    Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::EndOfSequence)) => {
                        match one_more_run_then_stop {
                            true => {
                                let created =
                                    match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                                        Ok(created) => created.as_secs(),
                                        Err(e) => {
                                            return Err(format!(
                                                "Failed to get the current time. {}",
                                                e.to_string()
                                            ));
                                        }
                                    };

                                let chat_completion_chunk = ChatCompletionChunk {
                                    id: "chatcmpl-123".to_string(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
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

                                if let Err(e) = graph.finish_single() {
                                    println!("Error: {:?}", &e);
                                    return Err(e.to_string());
                                }

                                one_more_run_then_stop = false;

                                // serialize chat completion chunk
                                let chunk = match serde_json::to_string(&chat_completion_chunk) {
                                    Ok(chunk) => chunk,
                                    Err(e) => {
                                        return Err(format!(
                                            "Fail to serialize chat completion chunk. {}",
                                            e.to_string()
                                        ));
                                    }
                                };

                                Ok(chunk)
                            }
                            false => Ok("[GGML] End of sequence".to_string()),
                        }
                    }
                    Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::ContextFull)) => {
                        match one_more_run_then_stop {
                            true => {
                                println!(
                                    "\n\n[WARNING] The generated message is cut off as the max context size is reached. If you'd like to get the complete answer, please increase the context size by setting the `--ctx-size` command option with a larger value, and then ask the same question again.\n"
                                );

                                let created =
                                    match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                                        Ok(created) => created.as_secs(),
                                        Err(e) => {
                                            return Err(format!(
                                                "Failed to get the current time. {}",
                                                e.to_string()
                                            ));
                                        }
                                    };

                                let chat_completion_chunk = ChatCompletionChunk {
                                    id: "chatcmpl-123".to_string(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model.clone(),
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
                                };

                                if let Err(e) = graph.finish_single() {
                                    println!("Error: {:?}", &e);
                                    return Err(e.to_string());
                                }

                                one_more_run_then_stop = false;

                                // serialize chat completion chunk
                                let chunk = match serde_json::to_string(&chat_completion_chunk) {
                                    Ok(chunk) => chunk,
                                    Err(e) => {
                                        return Err(format!(
                                            "Fail to serialize chat completion chunk. {}",
                                            e.to_string()
                                        ));
                                    }
                                };

                                Ok(chunk)
                            }
                            false => Ok("[GGML] End of sequence".to_string()),
                        }
                    }
                    Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::PromptTooLong)) => {
                        match one_more_run_then_stop {
                            true => {
                                println!("\n\n[WARNING] The prompt is too long. Please reduce the length of your input and try again.\n");

                                let created =
                                    match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                                        Ok(created) => created.as_secs(),
                                        Err(e) => {
                                            return Err(format!(
                                                "Failed to get the current time. {}",
                                                e.to_string()
                                            ));
                                        }
                                    };

                                let chat_completion_chunk = ChatCompletionChunk {
                                    id: "chatcmpl-123".to_string(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
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

                                one_more_run_then_stop = false;

                                // serialize chat completion chunk
                                let chunk = match serde_json::to_string(&chat_completion_chunk) {
                                    Ok(chunk) => chunk,
                                    Err(e) => {
                                        return Err(format!(
                                            "Fail to serialize chat completion chunk. {}",
                                            e.to_string()
                                        ));
                                    }
                                };

                                Ok(chunk)
                            }
                            false => Ok("[GGML] End of sequence".to_string()),
                        }
                    }
                    Err(e) => {
                        println!("Error: {:?}", &e);
                        return Err(e.to_string());
                    }
                }
            });

            // create hyer stream
            let stream = stream
                .try_take_while(|x| future::ready(Ok(x != "[GGML] End of sequence" && x != "")));

            Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .body(Body::wrap_stream(stream))
        }
        Some(false) | None => {
            match graph.compute() {
                Ok(_) => {
                    // Retrieve the output.
                    let max_buffer_size = match MAX_BUFFER_SIZE.get() {
                        Some(max_buffer_size) => max_buffer_size,
                        None => {
                            return error::internal_server_error(String::from(
                                "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
                            ));
                        }
                    };
                    let mut output_buffer = vec![0u8; *max_buffer_size];
                    let mut output_size = match graph.get_output(0, &mut output_buffer) {
                        Ok(size) => size,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Fail to get output tensor: {msg}",
                                msg = e.to_string()
                            ));
                        }
                    };
                    output_size = std::cmp::min(*max_buffer_size, output_size);

                    // convert inference result to string
                    let output = match std::str::from_utf8(&output_buffer[..output_size]) {
                        Ok(output) => output,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to decode the result bytes to a utf-8 string. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // post-process
                    let message = match post_process(&output, template_ty) {
                        Ok(message) => message,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to post-process the output. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // retrieve the number of prompt and completion tokens
                    let token_info = match get_token_info(&graph) {
                        Ok(token_info) => token_info,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to get the number of prompt and completion tokens. {}",
                                e.to_string()
                            ));
                        }
                    };
                    if log_prompts {
                        print_log_begin_separator("PROMPT (Tokens)", Some("*"), None);
                        println!(
                            "\nprompt tokens: {}, completion_tokens: {}",
                            token_info.prompt_tokens, token_info.completion_tokens
                        );
                        print_log_end_separator(Some("*"), None);
                    }

                    let created = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                        Ok(created) => created.as_secs(),
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to get the current time. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // create ChatCompletionResponse
                    let chat_completion_obejct = ChatCompletionObject {
                        id: uuid::Uuid::new_v4().to_string(),
                        object: String::from("chat.completion"),
                        created,
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

                    let s = match serde_json::to_string(&chat_completion_obejct) {
                        Ok(s) => s,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Fail to serialize chat completion object. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // return response
                    Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .body(Body::from(s))
                }
                Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::ContextFull)) => {
                    println!(
                        "\n\n[WARNING] The generated message is cut off as the max context size is reached. If you'd like to get the complete answer, please increase the context size by setting the `--ctx-size` command option with a larger value, and then ask the same question again.\n"
                    );

                    // Retrieve the output.
                    let max_buffer_size = match MAX_BUFFER_SIZE.get() {
                        Some(max_buffer_size) => max_buffer_size,
                        None => {
                            return error::internal_server_error(String::from(
                                "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
                            ));
                        }
                    };
                    let mut output_buffer = vec![0u8; *max_buffer_size];
                    let mut output_size = match graph.get_output(0, &mut output_buffer) {
                        Ok(size) => size,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Fail to get output tensor: {msg}",
                                msg = e.to_string()
                            ));
                        }
                    };
                    output_size = std::cmp::min(*max_buffer_size, output_size);

                    // convert inference result to string
                    let output = match std::str::from_utf8(&output_buffer[..output_size]) {
                        Ok(output) => output,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // post-process
                    let message = match post_process(&output, template_ty) {
                        Ok(message) => message,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to post-process the output. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // retrieve the number of prompt and completion tokens
                    let token_info = match get_token_info(&graph) {
                        Ok(token_info) => token_info,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to get the number of prompt and completion tokens. {}",
                                e.to_string()
                            ));
                        }
                    };
                    if log_prompts {
                        print_log_begin_separator("PROMPT (Tokens)", Some("*"), None);
                        println!(
                            "\nprompt tokens: {}, completion_tokens: {}",
                            token_info.prompt_tokens, token_info.completion_tokens
                        );
                        print_log_end_separator(Some("*"), None);
                    }

                    let created = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                        Ok(created) => created.as_secs(),
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to get the current time. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // create ChatCompletionResponse
                    let chat_completion_obejct = ChatCompletionObject {
                        id: uuid::Uuid::new_v4().to_string(),
                        object: String::from("chat.completion"),
                        created,
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

                    // serialize chat completion object
                    let s = match serde_json::to_string(&chat_completion_obejct) {
                        Ok(s) => s,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Fail to serialize chat completion object. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // return response
                    Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .body(Body::from(s))
                }
                Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::PromptTooLong)) => {
                    println!("\n\n[WARNING] The prompt is too long. Please reduce the length of your input and try again.\n");

                    // Retrieve the output.
                    let max_buffer_size = match MAX_BUFFER_SIZE.get() {
                        Some(max_buffer_size) => max_buffer_size,
                        None => {
                            return error::internal_server_error(String::from(
                                "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
                            ));
                        }
                    };
                    let mut output_buffer = vec![0u8; *max_buffer_size];
                    let mut output_size = match graph.get_output(0, &mut output_buffer) {
                        Ok(size) => size,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Fail to get output tensor: {msg}",
                                msg = e.to_string()
                            ));
                        }
                    };
                    output_size = std::cmp::min(*max_buffer_size, output_size);

                    // convert inference result to string
                    let output = match std::str::from_utf8(&output_buffer[..output_size]) {
                        Ok(output) => output,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // post-process
                    let message = match post_process(output, template_ty) {
                        Ok(message) => message,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to post-process the output. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // retrieve the number of prompt and completion tokens
                    let token_info = match get_token_info(&graph) {
                        Ok(token_info) => token_info,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to get the number of prompt and completion tokens. {}",
                                e.to_string()
                            ));
                        }
                    };
                    if log_prompts {
                        print_log_begin_separator("PROMPT (Tokens)", Some("*"), None);
                        println!(
                            "\nprompt tokens: {}, completion_tokens: {}",
                            token_info.prompt_tokens, token_info.completion_tokens
                        );
                        print_log_end_separator(Some("*"), None);
                    }

                    let created = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                        Ok(created) => created.as_secs(),
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Failed to get the current time. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // create ChatCompletionResponse
                    let chat_completion_obejct = ChatCompletionObject {
                        id: uuid::Uuid::new_v4().to_string(),
                        object: String::from("chat.completion"),
                        created,
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

                    // serialize chat completion object
                    let s = match serde_json::to_string(&chat_completion_obejct) {
                        Ok(s) => s,
                        Err(e) => {
                            return error::internal_server_error(format!(
                                "Fail to serialize chat completion object. {}",
                                e.to_string()
                            ));
                        }
                    };

                    // return response
                    Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .body(Body::from(s))
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
    let max_buffer_size = match MAX_BUFFER_SIZE.get() {
        Some(max_buffer_size) => max_buffer_size,
        None => {
            return Err(String::from(
                "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
            ))
        }
    };
    let mut output_buffer = vec![0u8; *max_buffer_size];
    let mut output_size = match graph.get_output(0, &mut output_buffer) {
        Ok(size) => size,
        Err(e) => {
            return Err(format!(
                "Fail to get output tensor: {msg}",
                msg = e.to_string()
            ))
        }
    };
    output_size = std::cmp::min(*max_buffer_size, output_size);

    Ok(output_buffer[..output_size].to_vec())
}

fn update_metadata(
    chat_request: &ChatCompletionRequest,
    available_completion_tokens: u64,
) -> Result<(), String> {
    let mut should_update = false;
    let mut metadata = match METADATA.get() {
        Some(metadata) => metadata.clone(),
        None => {
            return Err(String::from(
                "Fail to get the underlying value of `METADATA`.",
            ));
        }
    };

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
                return Err(format!(
                    "Fail to serialize metadata to a JSON string. {}",
                    e.to_string()
                ));
            }
        };

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

        // update metadata
        if graph
            .set_input(1, wasi_nn::TensorType::U8, &[1], config.as_bytes())
            .is_err()
        {
            return Err(String::from("Fail to update metadata"));
        }
    }

    Ok(())
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
                            return Ok((prompt, ctx_size - max_prompt_tokens));
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

fn get_token_info(graph: &Graph) -> Result<TokenInfo, String> {
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

struct TokenInfo {
    prompt_tokens: u64,
    completion_tokens: u64,
}
