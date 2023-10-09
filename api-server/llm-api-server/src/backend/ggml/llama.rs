use crate::error;
use hyper::{body::to_bytes, Body, Request, Response};
use prompt::BuildPrompt;
use xin::{
    chat::{
        ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseMessage,
        ChatCompletionRole, FinishReason,
    },
    common::Usage,
};

pub(crate) async fn llama_models_handler() -> Result<Response<Body>, hyper::Error> {
    println!("llama_models_handler not implemented");
    error::not_implemented()
}

pub(crate) async fn llama_embeddings_handler() -> Result<Response<Body>, hyper::Error> {
    println!("llama_embeddings_handler not implemented");
    error::not_implemented()
}

pub(crate) async fn llama_completions_handler() -> Result<Response<Body>, hyper::Error> {
    println!("llama_completions_handler not implemented");
    error::not_implemented()
}

/// Processes a chat-completion request and returns a chat-completion response with the answer from the model.
pub(crate) async fn llama_chat_completions_handler(
    mut req: Request<Body>,
    model_name: impl AsRef<str>,
) -> Result<Response<Body>, hyper::Error> {
    if req.method().eq(&hyper::http::Method::OPTIONS) {
        println!("[CHAT] Empty request received! Returns empty response!");

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

    println!("[CHAT] New chat begins ...");

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut chat_request: xin::chat::ChatCompletionRequest =
        serde_json::from_slice(&body_bytes).unwrap();

    // ! todo: according to the model name in the request, dynamically build the prompt
    // * build prompt for codellama
    // let prompt = match prompt::llama::CodeLlamaInstructPrompt::build(chat_request.messages.as_mut())
    // {
    //     Ok(prompt) => prompt,
    //     Err(e) => {
    //         return error::internal_server_error(e.to_string());
    //     }
    // };
    // * build prompt for mistral
    // if chat_request.messages[0].role == ChatCompletionRole::System {
    //     chat_request.messages.remove(0);
    // }
    // let prompt = match prompt::mistral::MistralInstructPrompt::build(chat_request.messages.as_mut())
    // {
    //     Ok(prompt) => prompt,
    //     Err(e) => {
    //         return error::internal_server_error(e.to_string());
    //     }
    // };
    // * build prompt for llama2chat
    let prompt = match prompt::llama::Llama2ChatPrompt::build(chat_request.messages.as_mut()) {
        Ok(prompt) => prompt,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    // run inference
    let buffer = infer(model_name.as_ref(), prompt.trim()).await;

    // convert inference result to string
    let model_answer = String::from_utf8(buffer.clone()).unwrap();
    let assistant_message = model_answer.trim();

    println!("[CHAT] Bot answer: {}", assistant_message);

    println!("[CHAT] New chat ends.");

    // create ChatCompletionResponse
    let chat_completion_obejct = ChatCompletionResponse {
        id: String::new(),
        object: String::from("chat.completion"),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: chat_request.model.clone(),
        choices: vec![ChatCompletionResponseChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                role: ChatCompletionRole::Assistant,
                content: String::from(assistant_message),
                function_call: None,
            },
            finish_reason: FinishReason::stop,
        }],
        usage: Usage {
            prompt_tokens: 9,
            completion_tokens: 12,
            total_tokens: 21,
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
pub(crate) async fn infer(model_name: impl AsRef<str>, prompt: impl AsRef<str>) -> Vec<u8> {
    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::CPU)
            .build_from_cache(model_name.as_ref())
            .unwrap();
    // println!("Loaded model into wasi-nn with ID: {:?}", graph);

    let mut context = graph.init_execution_context().unwrap();
    // println!("Created wasi-nn execution context with ID: {:?}", context);

    let tensor_data = prompt.as_ref().trim().as_bytes().to_vec();
    // println!("Read input tensor, size in bytes: {}", tensor_data.len());
    context
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .unwrap();

    // Execute the inference.
    context.compute().unwrap();
    // println!("Executed model inference");

    // Retrieve the output.
    let mut output_buffer = vec![0u8; 2048];
    let size = context.get_output(0, &mut output_buffer).unwrap();
    output_buffer[..size].to_vec()
}
