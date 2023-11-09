mod backend;
mod error;

use chat_prompts::PromptTemplateType;
use clap::{Arg, ArgAction, Command};
use error::ServerError;
use hyper::{
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server,
};
use once_cell::sync::{Lazy, OnceCell};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, str::FromStr, sync::Mutex};

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

const DEFAULT_SOCKET_ADDRESS: &str = "0.0.0.0:8080";
const DEFAULT_CTX_SIZE: &str = "4096";

static CTX_SIZE: OnceCell<usize> = OnceCell::new();

static GRAPH: OnceCell<Mutex<wasi_nn::Graph>> = OnceCell::new();

#[derive(Clone, Debug)]
pub struct AppState {
    pub state_thing: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), ServerError> {
    let matches = Command::new("Llama API Server")
        .arg(
            Arg::new("socket_addr")
                .short('s')
                .long("socket-addr")
                .value_name("IP:PORT")
                .help("Sets the socket address")
                .default_value(DEFAULT_SOCKET_ADDRESS),
        )
        .arg(
            Arg::new("model_name")
                .short('m')
                .long("model-name")
                .value_name("MODEL-NAME")
                .help("Sets the model name")
                .default_value("default"),
        )
        .arg(
            Arg::new("model_alias")
                .short('a')
                .long("model-alias")
                .value_name("MODEL-ALIAS")
                .help("Sets the alias name of the model in WasmEdge runtime")
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
                .default_value("4096"),
        )
        .arg(
            Arg::new("reverse_prompt")
                .short('r')
                .long("reverse-prompt")
                .value_name("REVERSE_PROMPT")
                .help("Halt generation at PROMPT, return control."),
        )
        .arg(
            Arg::new("prompt_template")
                .short('p')
                .long("prompt-template")
                .value_parser([
                    "llama-2-chat",
                    "codellama-instruct",
                    "mistral-instruct-v0.1",
                    "mistrallite",
                    "openchat",
                    "belle-llama-2-chat",
                    "vicuna-chat",
                    "chatml",
                    "baichuan-2",
                ])
                .value_name("TEMPLATE")
                .help("Sets the prompt template.")
                .default_value("llama-2-chat"),
        )
        .arg(
            Arg::new("stream_stdout")
                .long("stream-stdout")
                .value_name("STREAM_STDOUT")
                .help("Print the output to stdout in the streaming way")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    // socket address
    let socket_addr = matches
        .get_one::<String>("socket_addr")
        .unwrap()
        .to_string();
    let addr: SocketAddr = match socket_addr.parse() {
        Ok(addr) => addr,
        Err(e) => {
            return Err(ServerError::SocketAddr(e.to_string()));
        }
    };
    println!(
        "[INFO] Socket address: {socket_addr}",
        socket_addr = socket_addr
    );

    // model name
    let model_name = matches.get_one::<String>("model_name").unwrap().to_string();
    println!("[INFO] Model name: {name}", name = &model_name);

    // model alias
    let model_alias = matches
        .get_one::<String>("model_alias")
        .unwrap()
        .to_string();
    println!("[INFO] Model alias: {alias}", alias = &model_alias);

    // create a `ModelInfo` instance
    let model_info = ModelInfo::new(model_name, model_alias);

    // create an `Options` instance
    let mut options = Options::default();

    // prompt context size
    let ctx_size = matches.get_one::<u32>("ctx_size").unwrap();
    if CTX_SIZE.set(*ctx_size as usize * 6).is_err() {
        return Err(ServerError::PromptContextSize);
    }
    println!("[INFO] Prompt context size: {size}", size = ctx_size);

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

    // type of prompt template
    let prompt_template = matches
        .get_one::<String>("prompt_template")
        .unwrap()
        .to_string();
    let template_ty = match PromptTemplateType::from_str(&prompt_template) {
        Ok(template) => template,
        Err(e) => {
            return Err(ServerError::InvalidPromptTemplateType(e.to_string()));
        }
    };
    println!("[INFO] Prompt template: {ty:?}", ty = &template_ty);
    let ref_template_ty = std::sync::Arc::new(template_ty);

    // stream stdout
    let stream_stdout = matches.get_flag("stream_stdout");
    println!("[INFO] Stream stdout: {enable}", enable = stream_stdout);
    options.stream_stdout = stream_stdout;

    // serialize options
    let metadata = match serde_json::to_string(&options) {
        Ok(metadata) => metadata,
        Err(e) => {
            return Err(ServerError::InternalServerError(format!(
                "Fail to serialize options: {msg}",
                msg = e.to_string()
            )))
        }
    };

    println!("[INFO] Starting server ...");

    // load the model into wasi-nn
    let graph = match wasi_nn::GraphBuilder::new(
        wasi_nn::GraphEncoding::Ggml,
        wasi_nn::ExecutionTarget::AUTO,
    )
    .build_from_cache(&model_info.alias)
    {
        Ok(graph) => graph,
        Err(e) => {
            return Err(ServerError::InternalServerError(format!(
                "Fail to load model into wasi-nn: {msg}",
                msg = e.to_string()
            )))
        }
    };

    if GRAPH.set(Mutex::new(graph)).is_err() {
        return Err(ServerError::InternalServerError(
            "The GRAPH has already been initialized".to_owned(),
        ));
    }

    // the timestamp when the server is created
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let ref_created = std::sync::Arc::new(created);

    let new_service = make_service_fn(move |_| {
        let model_info = model_info.clone();
        let prompt_template_ty = ref_template_ty.clone();
        let created = ref_created.clone();
        let metadata = metadata.clone();
        async {
            Ok::<_, Error>(service_fn(move |req| {
                handle_request(
                    req,
                    model_info.clone(),
                    *prompt_template_ty.clone(),
                    *created.clone(),
                    metadata.clone(),
                )
            }))
        }
    });

    let server = Server::bind(&addr).serve(new_service);

    println!("[INFO] Listening on http://{}", addr);
    match server.await {
        Ok(_) => Ok(()),
        Err(e) => Err(ServerError::InternalServerError(e.to_string())),
    }
}

async fn handle_request(
    req: Request<Body>,
    model_info: ModelInfo,
    template_ty: PromptTemplateType,
    created: u64,
    metadata: String,
) -> Result<Response<Body>, hyper::Error> {
    match req.uri().path() {
        "/echo" => {
            return Ok(Response::new(Body::from("echo test")));
        }
        _ => backend::handle_llama_request(req, model_info, template_ty, created, metadata).await,
    }
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
struct Options {
    #[serde(rename = "enable-log")]
    log_enable: bool,
    #[serde(rename = "stream-stdout")]
    stream_stdout: bool,
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ModelInfo {
    name: String,
    alias: String,
}
impl ModelInfo {
    fn new(name: impl AsRef<str>, alias: impl AsRef<str>) -> Self {
        Self {
            name: name.as_ref().to_string(),
            alias: alias.as_ref().to_string(),
        }
    }
}
