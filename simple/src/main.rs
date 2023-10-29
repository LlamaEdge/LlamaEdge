use chat_prompts::{
    chat::{BuildChatPrompt, ChatPrompt},
    PromptTemplateType,
};
use clap::{Arg, Command};
use endpoints::chat::{ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionRole};
use once_cell::sync::OnceCell;
use std::str::FromStr;

const DEFAULT_CTX_SIZE: &str = "2048";
static CTX_SIZE: OnceCell<usize> = OnceCell::new();

#[allow(unreachable_code)]
fn main() -> Result<(), String> {
    let matches = Command::new("Llama API Server")
        .arg(
            Arg::new("prompt_template")
                .short('p')
                .long("prompt-template")
                .value_parser([
                    "llama-2-chat",
                    "codellama-instruct",
                    "mistral-instruct-v0.1",
                    "belle-llama-2-chat",
                    "vicuna-chat",
                    "chatml",
                ])
                .value_name("TEMPLATE")
                .help("Sets the prompt template.")
                .required(true),
        )
        .arg(
            Arg::new("user_message")
                .short('u')
                .long("user-message")
                .value_name("MESSAGE")
                .help("Sets the user message.")
                .required(true),
        )
        .arg(
            Arg::new("model_alias")
                .short('m')
                .long("model-alias")
                .value_name("ALIAS")
                .help("Sets the model alias")
                .default_value("default"),
        )
        .arg(
            Arg::new("ctx_size")
                .short('c')
                .long("ctx-size")
                .value_parser(clap::value_parser!(u32))
                .value_name("CTX_SIZE")
                .help("Sets the prompt context size")
                .default_value(DEFAULT_CTX_SIZE),
        )
        .get_matches();

    // model alias
    let model_name = matches
        .get_one::<String>("model_alias")
        .unwrap()
        .to_string();
    println!("[INFO] Model alias: {alias}", alias = &model_name);

    // prompt context size
    let ctx_size = matches.get_one::<u32>("ctx_size").unwrap();
    if CTX_SIZE.set(*ctx_size as usize).is_err() {
        return Err(String::from("Fail to parse prompt context size"));
    }
    println!("[INFO] Prompt context size: {size}", size = ctx_size);

    // type of prompt template
    let prompt_template = matches
        .get_one::<String>("prompt_template")
        .unwrap()
        .to_string();
    let template_ty = match PromptTemplateType::from_str(&prompt_template) {
        Ok(template) => template,
        Err(e) => {
            return Err(format!(
                "Fail to parse prompt template type: {msg}",
                msg = e.to_string()
            ))
        }
    };
    println!("[INFO] Prompt template: {ty:?}", ty = &template_ty);

    // user message
    let user_message = matches
        .get_one::<String>("user_message")
        .unwrap()
        .to_string();

    // create prompt template
    let template = create_prompt_template(template_ty);

    // build chat request
    let mut chat_request = ChatCompletionRequest::default();
    chat_request
        .messages
        .push(ChatCompletionRequestMessage::new(
            ChatCompletionRole::User,
            user_message,
        ));

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

    println!("*** [prompt begin] ***");
    println!("{}", &prompt);
    println!("*** [prompt end] ***");

    // execute the inference
    let output = infer(model_name, &prompt)?;

    println!("\n[ASSISTANT] {}", output);

    Ok(())
}

fn infer(model_name: impl AsRef<str>, prompt: impl AsRef<str>) -> Result<String, String> {
    // load the model into wasi-nn
    let graph = match wasi_nn::GraphBuilder::new(
        wasi_nn::GraphEncoding::Ggml,
        wasi_nn::ExecutionTarget::CPU,
    )
    .build_from_cache(model_name.as_ref())
    {
        Ok(graph) => graph,
        Err(e) => {
            return Err(format!(
                "Fail to load model into wasi-nn: {msg}",
                msg = e.to_string()
            ))
        }
    };

    // initialize the execution context
    let mut context = match graph.init_execution_context() {
        Ok(context) => context,
        Err(e) => {
            return Err(format!(
                "Fail to create wasi-nn execution context: {msg}",
                msg = e.to_string()
            ))
        }
    };

    // read input tensor
    let tensor_data = prompt.as_ref().trim().as_bytes().to_vec();
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

    // retrieve the output
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
    let output = String::from_utf8_lossy(&output_buffer[..output_size]).to_string();

    Ok(output.trim().to_string())
}

fn create_prompt_template(template_ty: PromptTemplateType) -> ChatPrompt {
    match template_ty {
        PromptTemplateType::Llama2Chat => {
            ChatPrompt::Llama2ChatPrompt(chat_prompts::chat::llama::Llama2ChatPrompt::default())
        }
        PromptTemplateType::MistralInstructV01 => ChatPrompt::MistralInstructPrompt(
            chat_prompts::chat::mistral::MistralInstructPrompt::default(),
        ),
        PromptTemplateType::CodeLlama => ChatPrompt::CodeLlamaInstructPrompt(
            chat_prompts::chat::llama::CodeLlamaInstructPrompt::default(),
        ),
        PromptTemplateType::BelleLlama2Chat => ChatPrompt::BelleLlama2ChatPrompt(
            chat_prompts::chat::belle::BelleLlama2ChatPrompt::default(),
        ),
        PromptTemplateType::VicunaChat => {
            ChatPrompt::VicunaChatPrompt(chat_prompts::chat::vicuna::VicunaChatPrompt::default())
        }
        PromptTemplateType::ChatML => {
            ChatPrompt::ChatMLPrompt(chat_prompts::chat::chatml::ChatMLPrompt::default())
        }
    }
}
