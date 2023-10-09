use hyper::{body::to_bytes, Body, Request, Response};
use xin::{
    chat::{
        ChatCompletionRequestMessage, ChatCompletionResponse, ChatCompletionResponseChoice,
        ChatCompletionResponseMessage, ChatCompletionRole, FinishReason,
    },
    common::Usage,
};

pub(crate) async fn llama_models_handler() -> Result<Response<Body>, hyper::Error> {
    unimplemented!("llama_models_handler not implemented")
}

pub(crate) async fn llama_embeddings_handler() -> Result<Response<Body>, hyper::Error> {
    unimplemented!("llama_embeddings_handler not implemented")
}

pub(crate) async fn llama_completions_handler() -> Result<Response<Body>, hyper::Error> {
    unimplemented!("llama_completions_handler not implemented")
}

pub(crate) async fn llama_chat_completions_handler(
    mut req: Request<Body>,
    model_name: impl AsRef<str>,
) -> Result<Response<Body>, hyper::Error> {
    println!("\n============ Start of one-turn chat ============\n");

    if req.method().eq(&hyper::http::Method::OPTIONS) {
        println!("*** empty request, return empty response ***");

        let response = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .body(Body::empty())
            .unwrap();

        return Ok(response);
    }

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut chat_request: xin::chat::ChatCompletionRequest =
        serde_json::from_slice(&body_bytes).unwrap();

    // build prompt
    let prompt = build_prompt(chat_request.messages.as_mut());

    let buffer = infer(model_name.as_ref(), prompt.trim()).await;
    let model_answer = String::from_utf8(buffer.clone()).unwrap();
    let assistant_message = model_answer.trim();

    dbg!(assistant_message);

    // prepare ChatCompletionResponse
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

    let response = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        // .body(Body::from(buffer))
        .body(Body::from(
            serde_json::to_string(&chat_completion_obejct).unwrap(),
        ))
        .unwrap();

    println!("============ End of one-turn chat ============\n\n");

    Ok(response)
}

fn build_prompt(messages: &mut Vec<ChatCompletionRequestMessage>) -> String {
    if messages.len() == 0 {
        return String::new();
    }

    let system_message = messages.remove(0);
    let _system_prompt = create_system_prompt(&system_message);

    // ! debug
    let system_prompt = String::from("<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>");

    let mut prompt = String::new();

    assert!(messages.len() >= 1);

    // process the chat history
    for message in messages {
        if message.role == ChatCompletionRole::User {
            prompt = create_user_prompt(&prompt, &system_prompt, message.content.as_str());
        } else if message.role == ChatCompletionRole::Assistant {
            prompt = create_assistant_prompt(&prompt, message.content.as_str());
        }
    }

    println!("*** [prompt begin] ***");
    println!("{}", &prompt);
    println!("*** [prompt end] ***");

    prompt
}

/// Create a system prompt from a chat completion request message.
fn create_system_prompt(system_message: &ChatCompletionRequestMessage) -> String {
    format!(
        "<<SYS>>\n{content} <</SYS>>",
        content = system_message.content.as_str()
    )
}

/// Create a user prompt from a chat completion request message.
fn create_user_prompt(
    chat_history: impl AsRef<str>,
    system_prompt: impl AsRef<str>,
    content: impl AsRef<str>,
) -> String {
    match chat_history.as_ref().is_empty() {
        true => format!(
            "<s>[INST] {system_prompt}\n\n{user_message} [/INST]",
            system_prompt = system_prompt.as_ref(),
            user_message = content.as_ref().trim(),
        ),
        false => format!(
            "{chat_history}<s>[INST] {user_message} [/INST]",
            chat_history = chat_history.as_ref(),
            user_message = content.as_ref().trim(),
        ),
    }
}

/// create an assistant prompt from a chat completion request message.
fn create_assistant_prompt(chat_history: impl AsRef<str>, content: impl AsRef<str>) -> String {
    format!(
        "{prompt} {assistant_message} </s>",
        prompt = chat_history.as_ref(),
        assistant_message = content.as_ref().trim(),
    )
}

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
