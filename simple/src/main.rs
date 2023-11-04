use clap::{Arg, ArgAction, Command};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};

const DEFAULT_CTX_SIZE: &str = "4096";
static CTX_SIZE: OnceCell<usize> = OnceCell::new();

fn main() -> Result<(), String> {
    let matches = Command::new("Llama API Server")
        .arg(
            Arg::new("prompt")
                .short('p')
                .long("prompt")
                .value_name("PROMPT")
                .help("Sets the prompt string, including system message if required.")
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
        .arg(
            Arg::new("n_predict")
                .short('n')
                .long("n-predict")
                .value_parser(clap::value_parser!(u32))
                .value_name("N_PRDICT")
                .help("Number of tokens to predict")
                .default_value("1024"),
        )
        .arg(
            Arg::new("n_gpu_layers")
                .short('g')
                .long("n-gpu-layers")
                .value_parser(clap::value_parser!(u32))
                .value_name("N_GPU_LAYERS")
                .help("Number of layers to run on the GPU")
                .default_value("100"),
        )
        .arg(
            Arg::new("batch_size")
                .short('b')
                .long("batch-size")
                .value_parser(clap::value_parser!(u32))
                .value_name("BATCH_SIZE")
                .help("Batch size for prompt processing")
                .default_value("512"),
        )
        .arg(
            Arg::new("reverse_prompt")
                .short('r')
                .long("reverse-prompt")
                .value_name("REVERSE_PROMPT")
                .help("Halt generation at PROMPT, return control."),
        )
        .arg(
            Arg::new("log_enable")
                .long("log-enable")
                .value_name("LOG_ENABLE")
                .help("Enable trace logs")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    // model alias
    let model_name = matches
        .get_one::<String>("model_alias")
        .unwrap()
        .to_string();

    // prompt
    let prompt = matches.get_one::<String>("prompt").unwrap().to_string();

    // create an `Options` instance
    let mut options = Options::default();

    // prompt context size
    let ctx_size = matches.get_one::<u32>("ctx_size").unwrap();
    CTX_SIZE
        .set(*ctx_size as usize * 6)
        .expect("Fail to parse prompt context size");
    println!("[INFO] prompt context size: {size}", size = ctx_size);

    // number of tokens to predict
    let n_predict = matches.get_one::<u32>("n_predict").unwrap();
    println!("[INFO] Number of tokens to predict: {n}", n = n_predict);
    options.n_predict = *n_predict as u64;

    // n_gpu_layers
    let n_gpu_layers = matches.get_one::<u32>("n_gpu_layers").unwrap();
    println!(
        "[INFO] Number of layers to run on the GPU: {n}",
        n = n_gpu_layers
    );
    options.n_gpu_layers = *n_gpu_layers as u64;

    // batch size
    let batch_size = matches.get_one::<u32>("batch_size").unwrap();
    println!(
        "[INFO] Batch size for prompt processing: {size}",
        size = batch_size
    );
    options.batch_size = *batch_size as u64;

    // reverse_prompt
    if let Some(reverse_prompt) = matches.get_one::<String>("reverse_prompt") {
        println!("[INFO] Reverse prompt: {prompt}", prompt = &reverse_prompt);
        options.reverse_prompt = Some(reverse_prompt.to_string());
    }

    // log
    let log_enable = matches.get_flag("log_enable");
    println!("[INFO] Log enable: {enable}", enable = log_enable);
    options.log_enable = log_enable;

    // load the model into wasi-nn
    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::AUTO)
            .build_from_cache(&model_name)
            .expect("Failed to load the model");

    // initialize the execution context
    let mut context = graph
        .init_execution_context()
        .expect("Failed to init context");

    // set metadata
    let metadata = serde_json::to_string(&options).expect("Fail to serialize options");
    context
        .set_input(
            1,
            wasi_nn::TensorType::U8,
            &[1],
            metadata.as_bytes().to_owned(),
        )
        .expect("Fail to set metadata");

    // set input tensor
    let tensor_data = prompt.as_str().as_bytes().to_vec();
    context
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set prompt as the input tensor");

    // execute the inference
    context.compute().expect("Failed to complete inference");

    // retrieve the output
    let mut output_buffer = vec![0u8; *CTX_SIZE.get().unwrap()];
    let mut output_size = context
        .get_output(0, &mut output_buffer)
        .expect("Failed to get output tensor");
    output_size = std::cmp::min(*CTX_SIZE.get().unwrap(), output_size);
    let output = String::from_utf8_lossy(&output_buffer[..output_size]).to_string();

    println!("\nprompt: {}", &prompt);
    println!("\noutput: {}", output);

    Ok(())
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
    #[serde(skip_serializing_if = "Option::is_none", rename = "reverse-prompt")]
    reverse_prompt: Option<String>,
}
