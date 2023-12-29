mod error;

use chat_prompts::{
    chat::{BuildChatPrompt, ChatPrompt},
    PromptTemplateType,
};
use clap::{crate_version, Arg, ArgAction, Command};
use endpoints::chat::{ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionRole};
use error::ChatError;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Write;
use std::str::FromStr;

static MAX_BUFFER_SIZE: OnceCell<usize> = OnceCell::new();

#[allow(unreachable_code)]
fn main() -> Result<(), String> {
    let matches = Command::new("llama-chat")
        .version(crate_version!())
        .arg(
            Arg::new("model_alias")
                .short('a')
                .long("model-alias")
                .value_name("ALIAS")
                .help("Model alias")
                .default_value("default"),
        )
        .arg(
            Arg::new("ctx_size")
                .short('c')
                .long("ctx-size")
                .value_parser(clap::value_parser!(u64))
                .value_name("CTX_SIZE")
                .help("Size of the prompt context")
                .default_value("512"),
        )
        .arg(
            Arg::new("n_predict")
                .short('n')
                .long("n-predict")
                .value_parser(clap::value_parser!(u64))
                .value_name("N_PRDICT")
                .help("Number of tokens to predict")
                .default_value("1024"),
        )
        .arg(
            Arg::new("n_gpu_layers")
                .short('g')
                .long("n-gpu-layers")
                .value_parser(clap::value_parser!(u64))
                .value_name("N_GPU_LAYERS")
                .help("Number of layers to run on the GPU")
                .default_value("100"),
        )
        .arg(
            Arg::new("batch_size")
                .short('b')
                .long("batch-size")
                .value_parser(clap::value_parser!(u64))
                .value_name("BATCH_SIZE")
                .help("Batch size for prompt processing")
                .default_value("512"),
        )
        .arg(
            Arg::new("temp")
                .long("temp")
                .value_parser(clap::value_parser!(f32))
                .value_name("TEMP")
                .help("Temperature for sampling")
                .default_value("0.8"),
        )
        .arg(
            Arg::new("repeat_penalty")
                .long("repeat-penalty")
                .value_parser(clap::value_parser!(f32))
                .value_name("REPEAT_PENALTY")
                .help("Penalize repeat sequence of tokens")
                .default_value("1.1"),
        )
        .arg(
            Arg::new("reverse_prompt")
                .short('r')
                .long("reverse-prompt")
                .value_name("REVERSE_PROMPT")
                .help("Halt generation at PROMPT, return control."),
        )
        .arg(
            Arg::new("system_prompt")
                .short('s')
                .long("system-prompt")
                .value_name("SYSTEM_PROMPT")
                .help("System prompt message string")
                .default_value("[Default system message for the prompt template]"),
        )
        .arg(
            Arg::new("prompt_template")
                .short('p')
                .long("prompt-template")
                .value_parser([
                    "llama-2-chat",
                    "codellama-instruct",
                    "mistral-instruct-v0.1",
                    "mistral-instruct",
                    "mistrallite",
                    "openchat",
                    "belle-llama-2-chat",
                    "vicuna-chat",
                    "vicuna-1.1-chat",
                    "chatml",
                    "baichuan-2",
                    "wizard-coder",
                    "zephyr",
                    "intel-neural",
                    "deepseek-chat",
                    "deepseek-coder",
                    "solar-instruct",
                    "phi-2-chat",
                    "phi-2-instruct",
                ])
                .value_name("TEMPLATE")
                .help("Prompt template.")
                .default_value("llama-2-chat"),
        )
        .arg(
            Arg::new("log_prompts")
                .long("log-prompts")
                .value_name("LOG_PROMPTS")
                .help("Print prompt strings to stdout")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log_stat")
                .long("log-stat")
                .value_name("LOG_STAT")
                .help("Print statistics to stdout")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log_all")
                .long("log-all")
                .value_name("LOG_all")
                .help("Print all log information to stdout")
                .action(ArgAction::SetTrue),
        )
        .after_help("Example: the command to run `llama-2-7B` model,\n  wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-chat.wasm -p llama-2-chat\n")
        .get_matches();

    // create an `Options` instance
    let mut options = Options::default();

    // model alias
    let model_name = matches
        .get_one::<String>("model_alias")
        .unwrap()
        .to_string();
    println!("[INFO] Model alias: {alias}", alias = &model_name);

    // prompt context size
    let ctx_size = matches.get_one::<u64>("ctx_size").unwrap();
    println!("[INFO] Prompt context size: {size}", size = ctx_size);
    options.ctx_size = *ctx_size;

    // max buffer size
    if MAX_BUFFER_SIZE.set(*ctx_size as usize).is_err() {
        return Err(String::from(
            "Fail to set `MAX_BUFFER_SIZE`. It is already set.",
        ));
    }

    // number of tokens to predict
    let n_predict = matches.get_one::<u64>("n_predict").unwrap();
    println!("[INFO] Number of tokens to predict: {n}", n = n_predict);
    options.n_predict = *n_predict;

    // n_gpu_layers
    let n_gpu_layers = matches.get_one::<u64>("n_gpu_layers").unwrap();
    println!(
        "[INFO] Number of layers to run on the GPU: {n}",
        n = n_gpu_layers
    );
    options.n_gpu_layers = *n_gpu_layers;

    // batch size
    let batch_size = matches.get_one::<u64>("batch_size").unwrap();
    println!(
        "[INFO] Batch size for prompt processing: {size}",
        size = batch_size
    );
    options.batch_size = *batch_size;

    // temperature
    let temp = matches.get_one::<f32>("temp").unwrap();
    println!("[INFO] Temperature for sampling: {temp}", temp = temp);
    options.temp = *temp;

    // repeat penalty
    let repeat_penalty = matches.get_one::<f32>("repeat_penalty").unwrap();
    println!(
        "[INFO] Penalize repeat sequence of tokens: {penalty}",
        penalty = repeat_penalty
    );
    options.repeat_penalty = *repeat_penalty;

    // reverse_prompt
    if let Some(reverse_prompt) = matches.get_one::<String>("reverse_prompt") {
        println!("[INFO] Reverse prompt: {prompt}", prompt = &reverse_prompt);
        options.reverse_prompt = Some(reverse_prompt.to_string());
    }

    // system prompt
    let system_prompt = matches
        .get_one::<String>("system_prompt")
        .unwrap()
        .to_string();
    let system_prompt = match system_prompt == "[Default system message for the prompt template]" {
        true => {
            println!("[INFO] Use default system prompt");
            String::new()
        }
        false => {
            println!(
                "[INFO] Use custom system prompt: {prompt}",
                prompt = &system_prompt
            );
            system_prompt
        }
    };

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

    // log prompts
    let log_prompts = matches.get_flag("log_prompts");
    println!("[INFO] Log prompts: {enable}", enable = log_prompts);

    // log statistics
    let log_stat = matches.get_flag("log_stat");
    println!("[INFO] Log statistics: {enable}", enable = log_stat);

    // log all
    let log_all = matches.get_flag("log_all");
    println!("[INFO] Log all information: {enable}", enable = log_all);

    // set `log_enable`
    if log_stat || log_all {
        options.log_enable = true;
    }

    let template = create_prompt_template(template_ty.clone());
    let mut chat_request = ChatCompletionRequest::default();
    // put system_prompt into the `messages` of chat_request
    if !system_prompt.is_empty() {
        chat_request
            .messages
            .push(ChatCompletionRequestMessage::new(
                ChatCompletionRole::System,
                system_prompt,
            ));
    }

    // serialize metadata
    let metadata = match serde_json::to_string(&options) {
        Ok(metadata) => metadata,
        Err(e) => {
            return Err(format!(
                "Fail to serialize options: {msg}",
                msg = e.to_string()
            ))
        }
    };

    if log_stat || log_all {
        print_log_begin_separator(
            "MODEL INFO (Load Model & Init Execution Context)",
            Some("*"),
            None,
        );
    }

    // load the model into wasi-nn
    let graph = match wasi_nn::GraphBuilder::new(
        wasi_nn::GraphEncoding::Ggml,
        wasi_nn::ExecutionTarget::AUTO,
    )
    .config(metadata)
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

    if log_stat || log_all {
        print_log_end_separator(Some("*"), None);
    }

    let readme = "
================================== Running in interactive mode. ===================================\n
    - Press [Ctrl+C] to interject at any time.
    - Press [Return] to end the input.
    - For multi-line inputs, end each line with '\\' and press [Return] to get another line.\n";

    println!("{}", readme);

    loop {
        println!("\n[You]: ");
        let user_message = read_input();

        // put the user message into the messages sequence of chat_request
        chat_request
            .messages
            .push(ChatCompletionRequestMessage::new(
                ChatCompletionRole::User,
                user_message,
            ));

        if log_stat || log_all {
            print_log_begin_separator("STATISTICS (Set Input)", Some("*"), None);
        }

        // build prompt
        let max_prompt_tokens = *ctx_size * 4 / 5;
        let prompt = match build_prompt(
            &template,
            &mut chat_request,
            &mut context,
            max_prompt_tokens,
        ) {
            Ok(prompt) => prompt,
            Err(e) => {
                return Err(format!(
                    "Fail to generate prompt. Reason: {msg}",
                    msg = e.to_string()
                ))
            }
        };

        if log_stat || log_all {
            print_log_end_separator(Some("*"), None);
        }

        if log_prompts || log_all {
            print_log_begin_separator("PROMPT", Some("*"), None);
            println!("{}", &prompt);
            print_log_end_separator(Some("*"), None);
        }

        println!("\n[Bot]:");

        if log_stat || log_all {
            print_log_begin_separator("STATISTICS (Compute)", Some("*"), None);
        }

        let start_time = std::time::Instant::now();

        // compute
        let result = match options.reverse_prompt {
            Some(ref reverse_prompt) => stream_compute(&mut context, Some(reverse_prompt.as_str())),
            None => stream_compute(&mut context, None),
        };

        let elapsed = start_time.elapsed();

        if log_stat || log_all {
            print_log_end_separator(Some("*"), None);
        }

        match result {
            Ok(completion_message) => {
                let token_info = get_token_info(&context);

                if log_prompts || log_stat || log_all {
                    print_log_begin_separator("STATISTICS", Some("*"), None);

                    println!("\nPrompt tokens: {}", token_info.input_tokens);
                    println!("\n*** Completion tokens: {}", token_info.output_tokens);
                    println!(
                        "\nTotal tokens: {}",
                        token_info.input_tokens + token_info.output_tokens
                    );
                    println!("\nElapsed time: {:?}", elapsed);
                    println!(
                        "\nTokens per second (tps): {}. Note that the tps data is computed in the streaming mode. For more accurate tps data, please compute it in the non-streaming mode.",
                        token_info.output_tokens as f64 / elapsed.as_secs_f64()
                    );

                    print_log_end_separator(Some("*"), None);
                }

                // put the assistant message into the message sequence of chat_request
                chat_request
                    .messages
                    .push(ChatCompletionRequestMessage::new(
                        ChatCompletionRole::Assistant,
                        completion_message,
                    ));

                // this is the required step. Otherwise, will get a cumulative number when retrieve the number of output tokens of each round
                context.fini_single().unwrap();
            }
            Err(ChatError::ContextFull(completion_message)) => {
                println!(
                    "\n\n[WARNING] The message is cut off as the max context size is reached. You can try to ask the same question again, or increase the context size via the `--ctx-size` command option."
                );

                let token_info = get_token_info(&context);

                if log_prompts || log_stat || log_all {
                    print_log_begin_separator("STATISTICS", Some("*"), None);

                    println!("\nPrompt tokens: {}", token_info.input_tokens);
                    println!("\n*** Completion tokens: {}", token_info.output_tokens);
                    println!(
                        "\nTotal tokens: {}",
                        token_info.input_tokens + token_info.output_tokens
                    );
                    println!("\nElapsed time: {:?}", elapsed);
                    println!(
                        "\nTokens per second (tps): {}. Note that the tps data is computed in the streaming mode. For more accurate tps data, please compute it in the non-streaming mode.",
                        token_info.output_tokens as f64 / elapsed.as_secs_f64()
                    );

                    print_log_end_separator(Some("*"), None);
                }

                // put the assistant message into the message sequence of chat_request
                chat_request
                    .messages
                    .push(ChatCompletionRequestMessage::new(
                        ChatCompletionRole::Assistant,
                        completion_message,
                    ));

                // this is the required step. Otherwise, will get a cumulative number when retrieve the number of output tokens of each round
                context.fini_single().unwrap();
            }
            Err(e) => {
                return Err(format!(
                    "Fail to compute. Reason: {msg}",
                    msg = e.to_string()
                ))
            }
        }
    }

    Ok(())
}

// For single line input, just press [Return] to end the input.
// For multi-line input, end your input with '\\' and press [Return].
//
// For example:
//  [You]:
//  what is the capital of France?[Return]
//
//  [You]:
//  Count the words in the following sentence: \[Return]
//  \[Return]
//  You can use Git to save new files and any changes to already existing files as a bundle of changes called a commit, which can be thought of as a “revision” to your project.[Return]
//
fn read_input() -> String {
    let mut answer = String::new();
    loop {
        let mut temp = String::new();
        std::io::stdin()
            .read_line(&mut temp)
            .expect("The read bytes are not valid UTF-8");

        if temp.ends_with("\\\n") {
            temp.pop();
            temp.pop();
            temp.push('\n');
            answer.push_str(&temp);
            continue;
        } else if temp.ends_with("\n") {
            answer.push_str(&temp);
            return answer;
        } else {
            return answer;
        }
    }
}

fn print_log_begin_separator(
    title: impl AsRef<str>,
    ch: Option<&str>,
    len: Option<usize>,
) -> usize {
    let title = format!(" [LOG: {}] ", title.as_ref());

    let total_len: usize = len.unwrap_or(100);
    let separator_len: usize = (total_len - title.len()) / 2;

    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push_str(&title);
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push_str("\n");
    println!("{}", separator);
    total_len
}

fn print_log_end_separator(ch: Option<&str>, len: Option<usize>) {
    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(len.unwrap_or(100)).as_str());
    separator.push_str("\n");
    println!("{}", separator);
}

fn create_prompt_template(template_ty: PromptTemplateType) -> ChatPrompt {
    match template_ty {
        PromptTemplateType::Llama2Chat => {
            ChatPrompt::Llama2ChatPrompt(chat_prompts::chat::llama::Llama2ChatPrompt::default())
        }
        PromptTemplateType::MistralInstruct => ChatPrompt::MistralInstructPrompt(
            chat_prompts::chat::mistral::MistralInstructPrompt::default(),
        ),
        PromptTemplateType::MistralLite => {
            ChatPrompt::MistralLitePrompt(chat_prompts::chat::mistral::MistralLitePrompt::default())
        }
        PromptTemplateType::OpenChat => {
            ChatPrompt::OpenChatPrompt(chat_prompts::chat::openchat::OpenChatPrompt::default())
        }
        PromptTemplateType::CodeLlama => ChatPrompt::CodeLlamaInstructPrompt(
            chat_prompts::chat::llama::CodeLlamaInstructPrompt::default(),
        ),
        PromptTemplateType::BelleLlama2Chat => ChatPrompt::BelleLlama2ChatPrompt(
            chat_prompts::chat::belle::BelleLlama2ChatPrompt::default(),
        ),
        PromptTemplateType::VicunaChat => {
            ChatPrompt::VicunaChatPrompt(chat_prompts::chat::vicuna::VicunaChatPrompt::default())
        }
        PromptTemplateType::Vicuna11Chat => {
            ChatPrompt::Vicuna11ChatPrompt(chat_prompts::chat::vicuna::Vicuna11ChatPrompt::default())
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
    }
}

fn _post_process(output: impl AsRef<str>, template_ty: PromptTemplateType) -> String {
    println!("[DEBUG] Post-processing ...");

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
    } else {
        output.as_ref().trim().to_owned()
    }
}

fn stream_compute(
    context: &mut wasi_nn::GraphExecutionContext,
    stop: Option<&str>,
) -> Result<String, ChatError> {
    let mut output = String::new();

    // Compute one token at a time, and get the token using the get_output_single().
    loop {
        match context.compute_single() {
            Ok(_) => {
                // Retrieve the output.
                let max_output_size = *MAX_BUFFER_SIZE.get().unwrap();
                let mut output_buffer = vec![0u8; max_output_size];
                let mut output_size = context.get_output_single(0, &mut output_buffer).unwrap();
                output_size = std::cmp::min(max_output_size, output_size);
                let token = String::from_utf8_lossy(&output_buffer[..output_size]).to_string();

                // remove the redundant characters at the beginning of each answer
                if output.is_empty() && (token == " " || token == "\n") {
                    continue;
                }

                // trigger the stop condition
                if stop.is_some() && stop == Some(token.trim()) {
                    break;
                }

                if output.is_empty() && token.starts_with(" ") {
                    print!("{}", token.trim_start());
                } else {
                    print!("{}", token);
                }
                std::io::stdout().flush().unwrap();

                output += &token;
            }
            Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::EndOfSequence)) => {
                // Retrieve the output.
                let max_output_size = *MAX_BUFFER_SIZE.get().unwrap();
                let mut output_buffer = vec![0u8; max_output_size];
                let mut output_size = context.get_output_single(0, &mut output_buffer).unwrap();
                output_size = std::cmp::min(max_output_size, output_size);
                let token = String::from_utf8_lossy(&output_buffer[..output_size]).to_string();

                // remove the redundant characters at the beginning of each answer
                if output.is_empty() && (token == " " || token == "\n") {
                    continue;
                }

                // trigger the stop condition
                if stop.is_some() && stop == Some(token.trim()) {
                    break;
                }

                if output.is_empty() && token.starts_with(" ") {
                    print!("{}", token.trim_start());
                } else {
                    print!("{}", token);
                }
                std::io::stdout().flush().unwrap();

                output += &token;
                break;
            }
            Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::PromptTooLong)) => {
                panic!("[ERROR] BackendError: PromptTooLong. This error should not be triggered.")
            }
            Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::ContextFull)) => {
                return Err(ChatError::ContextFull(output));
            }
            Err(err) => {
                return Err(ChatError::Operation(err.to_string()));
            }
        }
    }
    println!("");

    Ok(output)
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct Options {
    #[serde(rename = "enable-log")]
    log_enable: bool,
    #[serde(rename = "ctx-size")]
    ctx_size: u64,
    #[serde(rename = "n-predict")]
    n_predict: u64,
    #[serde(rename = "n-gpu-layers")]
    n_gpu_layers: u64,
    #[serde(rename = "batch-size")]
    batch_size: u64,
    #[serde(rename = "temp")]
    temp: f32,
    #[serde(rename = "repeat-penalty")]
    repeat_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none", rename = "reverse-prompt")]
    reverse_prompt: Option<String>,
}

fn build_prompt(
    template: &ChatPrompt,
    chat_request: &mut ChatCompletionRequest,
    context: &mut wasi_nn::GraphExecutionContext,
    max_prompt_tokens: u64,
) -> Result<String, String> {
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
        if context
            .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
            .is_err()
        {
            return Err(String::from("Fail to set input tensor"));
        };

        // Retrieve the number of prompt tokens.
        let max_input_size = *MAX_BUFFER_SIZE.get().unwrap();
        let mut input_buffer = vec![0u8; max_input_size];
        let mut input_size = context.get_output(1, &mut input_buffer).unwrap();
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

fn get_token_info(context: &wasi_nn::GraphExecutionContext) -> TokenInfo {
    let max_output_size = *MAX_BUFFER_SIZE.get().unwrap();
    let mut output_buffer = vec![0u8; max_output_size];
    let mut output_size = context.get_output(1, &mut output_buffer).unwrap();
    output_size = std::cmp::min(max_output_size, output_size);
    let token_info: Value = serde_json::from_slice(&output_buffer[..output_size]).unwrap();
    TokenInfo {
        input_tokens: token_info["input_tokens"].as_u64().unwrap(),
        output_tokens: token_info["output_tokens"].as_u64().unwrap(),
    }
}

struct TokenInfo {
    input_tokens: u64,
    output_tokens: u64,
}
