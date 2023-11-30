use crate::{error, ModelInfo, CTX_SIZE};
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
        ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice,
        ChatCompletionResponseMessage, ChatCompletionRole,
    },
    common::{FinishReason, Usage},
    completions::{CompletionChoice, CompletionObject, CompletionRequest},
    models::{ListModelsResponse, Model},
};
use hyper::{body::to_bytes, Body, Request, Response};
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
    metadata: String,
) -> Result<Response<Body>, hyper::Error> {
    println!("[COMPLETION] New completion begins ...");

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let completion_request: CompletionRequest = serde_json::from_slice(&body_bytes).unwrap();

    let prompt = completion_request.prompt.join(" ");

    // ! todo: a temp solution of computing the number of tokens in prompt
    let prompt_tokens = prompt.split_whitespace().count() as u32;

    let buffer = match infer(prompt.trim(), metadata).await {
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
    metadata: String,
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
            PromptTemplateType::MistralInstructV01 => {
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
        }
    }
    let template = create_prompt_template(template_ty);

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut chat_request: ChatCompletionRequest = serde_json::from_slice(&body_bytes).unwrap();

    // build prompt
    let prompt = match template.build(chat_request.messages.as_mut()) {
        Ok(prompt) => prompt,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    if log_prompts {
        println!("\n---------------- [LOG: PROMPT] ---------------------\n");
        println!("{}", &prompt);
        println!("\n----------------------------------------------------\n");
    }

    // ! todo: a temp solution of computing the number of tokens in prompt
    let prompt_tokens = prompt.split_whitespace().count() as u32;

    // run inference
    let buffer = match infer(prompt, metadata).await {
        Ok(buffer) => buffer,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    // convert inference result to string
    let output = String::from_utf8(buffer.clone()).unwrap();

    let message = post_process(&output, template_ty);

    // ! todo: a temp solution of computing the number of tokens in assistant_message
    let completion_tokens = message.split_whitespace().count() as u32;

    print(&message);

    // create ChatCompletionResponse
    let chat_completion_obejct = ChatCompletionResponse {
        id: uuid::Uuid::new_v4().to_string(),
        object: String::from("chat.completion"),
        created: SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: chat_request.model.clone().unwrap_or_default(),
        choices: vec![ChatCompletionResponseChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                role: ChatCompletionRole::Assistant,
                content: message,
                function_call: None,
            },
            finish_reason: FinishReason::stop,
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
            serde_json::to_string(&chat_completion_obejct).unwrap(),
        ));
    match result {
        Ok(response) => Ok(response),
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

/// Runs inference on the model with the given name and returns the output.
pub(crate) async fn infer(
    prompt: impl AsRef<str>,
    metadata: String,
) -> std::result::Result<Vec<u8>, String> {
    let graph = crate::GRAPH.get().unwrap().lock().unwrap();

    let mut context = graph.init_execution_context().unwrap();

    // set metadata
    if context
        .set_input(
            1,
            wasi_nn::TensorType::U8,
            &[1],
            metadata.as_bytes().to_owned(),
        )
        .is_err()
    {
        return Err(String::from("Fail to set metadata"));
    };

    let tensor_data = prompt.as_ref().as_bytes().to_vec();
    // println!("Read input tensor, size in bytes: {}", tensor_data.len());
    if context
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .is_err()
    {
        return Err(String::from("Fail to set input tensor"));
    };

    // execute the inference
    if context.compute().is_err() {
        return Err(String::from("Fail to execute model inference"));
    }

    // Retrieve the output.
    let mut output_buffer = vec![0u8; *CTX_SIZE.get().unwrap()];
    let mut output_size = match context.get_output(0, &mut output_buffer) {
        Ok(size) => size,
        Err(e) => {
            return Err(format!(
                "Fail to get output tensor: {msg}",
                msg = e.to_string()
            ))
        }
    };
    output_size = std::cmp::min(*CTX_SIZE.get().unwrap(), output_size);
    Ok(output_buffer[..output_size].to_vec())
}

fn post_process(output: impl AsRef<str>, template_ty: PromptTemplateType) -> String {
    if template_ty == PromptTemplateType::Baichuan2 {
        output.as_ref().split('\n').collect::<Vec<_>>()[0]
            .trim()
            .to_owned()
    } else if template_ty == PromptTemplateType::OpenChat {
        output
            .as_ref()
            .trim_end_matches("<|end_of_turn|>")
            .trim()
            .to_owned()
    } else if template_ty == PromptTemplateType::ChatML {
        if output.as_ref().contains("<|im_end|>") {
            output.as_ref().replace("<|im_end|>", "").trim().to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if template_ty == PromptTemplateType::Zephyr
        || template_ty == PromptTemplateType::MistralLite
    {
        if output.as_ref().contains("</s>") {
            output.as_ref().trim_end_matches("</s>").trim().to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else {
        output.as_ref().trim().to_owned()
    }
}

fn print(message: impl AsRef<str>) {
    println!("\n[ASSISTANT]:\n{}", message.as_ref().trim())
}
