mod backend;
mod error;
mod utils;

use anyhow::Result;
use chat_prompts::PromptTemplateType;
use clap::Parser;
use error::ServerError;
use hyper::{
    header,
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server, StatusCode,
};
use llama_core::MetadataBuilder;
use std::{net::SocketAddr, path::PathBuf};
use utils::log;

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

// default socket address of LlamaEdge API Server instance
const DEFAULT_SOCKET_ADDRESS: &str = "0.0.0.0:8080";

#[derive(Debug, Parser)]
#[command(author, about, version, long_about=None)]
struct Cli {
    /// Model name
    #[arg(short, long, default_value = "default")]
    model_name: String,
    /// Model alias
    #[arg(short = 'a', long, default_value = "default")]
    model_alias: String,
    /// Context size
    #[arg(short, long, default_value = "512")]
    ctx_size: u64,
    /// Sets the prompt template.
    #[arg(short, long, value_parser = clap::value_parser!(PromptTemplateType), required = true)]
    prompt_template: PromptTemplateType,
    /// Halt generation at PROMPT, return control.
    #[arg(short, long)]
    reverse_prompt: Option<String>,
    /// Number of tokens to predict
    #[arg(short, long, default_value = "1024")]
    n_predict: u64,
    /// Number of layers to run on the GPU
    #[arg(short = 'g', long, default_value = "100")]
    n_gpu_layers: u64,
    /// Batch size for prompt processing
    #[arg(short, long, default_value = "512")]
    batch_size: u64,
    /// Temperature for sampling
    #[arg(long, default_value = "1.0")]
    temp: f64,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 1.0 = disabled
    #[arg(long, default_value = "1.0")]
    top_p: f64,
    /// Penalize repeat sequence of tokens
    #[arg(long, default_value = "1.1")]
    repeat_penalty: f64,
    /// Repeat alpha presence penalty. 0.0 = disabled
    #[arg(long, default_value = "0.0")]
    presence_penalty: f64,
    /// Repeat alpha frequency penalty. 0.0 = disabled
    #[arg(long, default_value = "0.0")]
    frequency_penalty: f64,
    /// Path to the multimodal projector file
    #[arg(long)]
    llava_mmproj: Option<String>,
    /// Print prompt strings to stdout
    #[arg(long)]
    log_prompts: bool,
    /// Print statistics to stdout
    #[arg(long)]
    log_stat: bool,
    /// Print all log information to stdout
    #[arg(long)]
    log_all: bool,
    /// Socket address of LlamaEdge API Server instance
    #[arg(long, default_value = DEFAULT_SOCKET_ADDRESS)]
    socket_addr: String,
    /// Root path for the Web UI files
    #[arg(long, default_value = "chatbot-ui")]
    web_ui: PathBuf,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), ServerError> {
    let cli = Cli::parse();

    // log the version of the server
    log(format!(
        "\n[INFO] LlamaEdge version: {}",
        env!("CARGO_PKG_VERSION")
    ));

    // log the cli options
    log(format!("[INFO] Model name: {}", &cli.model_name));
    log(format!("[INFO] Model alias: {}", &cli.model_alias));
    log(format!("[INFO] Context size: {}", &cli.ctx_size));
    log(format!("[INFO] Prompt template: {}", &cli.prompt_template));
    if let Some(reverse_prompt) = &cli.reverse_prompt {
        log(format!("[INFO] reverse prompt: {}", reverse_prompt));
    }
    log(format!(
        "[INFO] Number of tokens to predict: {}",
        &cli.n_predict
    ));
    log(format!(
        "[INFO] Number of layers to run on the GPU: {}",
        &cli.n_gpu_layers
    ));
    log(format!(
        "[INFO] Batch size for prompt processing: {}",
        &cli.batch_size
    ));
    log(format!("[INFO] Temperature for sampling: {}", &cli.temp));
    log(format!(
        "[INFO] Top-p sampling (1.0 = disabled): {}",
        &cli.top_p
    ));
    log(format!(
        "[INFO] Penalize repeat sequence of tokens: {}",
        &cli.repeat_penalty
    ));
    log(format!(
        "[INFO] Presence penalty (0.0 = disabled): {}",
        &cli.presence_penalty
    ));
    log(format!(
        "[INFO] Frequency penalty (0.0 = disabled): {}",
        &cli.frequency_penalty
    ));
    if let Some(llava_mmproj) = &cli.llava_mmproj {
        log(format!("[INFO] Multimodal projector: {}", llava_mmproj));
    }
    log(format!("[INFO] Enable prompt log: {}", &cli.log_prompts));
    log(format!("[INFO] Enable plugin log: {}", &cli.log_stat));
    log(format!("[INFO] Socket address: {}", &cli.socket_addr));

    // create a Metadata instance
    let metadata = MetadataBuilder::new(cli.model_name, cli.model_alias, cli.prompt_template)
        .with_ctx_size(cli.ctx_size)
        .with_n_predict(cli.n_predict)
        .with_n_gpu_layers(cli.n_gpu_layers)
        .with_batch_size(cli.batch_size)
        .with_temperature(cli.temp)
        .with_top_p(cli.top_p)
        .with_repeat_penalty(cli.repeat_penalty)
        .with_presence_penalty(cli.presence_penalty)
        .with_frequency_penalty(cli.frequency_penalty)
        .with_reverse_prompt(cli.reverse_prompt)
        .with_mmproj(cli.llava_mmproj.clone())
        .enable_prompts_log(cli.log_prompts || cli.log_all)
        .enable_plugin_log(cli.log_stat || cli.log_all)
        .build();

    // initialize the core context
    llama_core::init_core_context(&[metadata])
        .map_err(|e| ServerError::Operation(format!("{}", e)))?;

    // get the plugin version info
    let plugin_info =
        llama_core::get_plugin_info().map_err(|e| ServerError::Operation(e.to_string()))?;
    log(format!(
        "[INFO] Wasi-nn-ggml plugin: b{build_number} (commit {commit_id})",
        build_number = plugin_info.build_number,
        commit_id = plugin_info.commit_id,
    ));

    let new_service = make_service_fn(move |_| {
        let web_ui = cli.web_ui.to_string_lossy().to_string();

        async move { Ok::<_, Error>(service_fn(move |req| handle_request(req, web_ui.clone()))) }
    });

    // socket address
    let addr = cli
        .socket_addr
        .parse::<SocketAddr>()
        .map_err(|e| ServerError::SocketAddr(e.to_string()))?;
    let server = Server::bind(&addr).serve(new_service);

    log(format!(
        "[INFO] LlamaEdge API server listening on http://{}:{}",
        addr.ip(),
        addr.port()
    ));

    match server.await {
        Ok(_) => Ok(()),
        Err(e) => Err(ServerError::Operation(e.to_string())),
    }
}

async fn handle_request(
    req: Request<Body>,
    web_ui: String,
) -> Result<Response<Body>, hyper::Error> {
    let path_str = req.uri().path();
    let path_buf = PathBuf::from(path_str);
    let mut path_iter = path_buf.iter();
    path_iter.next(); // Must be Some(OsStr::new(&path::MAIN_SEPARATOR.to_string()))
    let root_path = path_iter.next().unwrap_or_default();
    let root_path = "/".to_owned() + root_path.to_str().unwrap_or_default();

    match root_path.as_str() {
        "/echo" => Ok(Response::new(Body::from("echo test"))),
        "/v1" => backend::handle_llama_request(req).await,
        _ => Ok(static_response(path_str, web_ui)),
    }
}

fn static_response(path_str: &str, root: String) -> Response<Body> {
    let path = match path_str {
        "/" => "/index.html",
        _ => path_str,
    };

    let mime = mime_guess::from_path(path);

    match std::fs::read(format!("{root}/{path}")) {
        Ok(content) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime.first_or_text_plain().to_string())
            .body(Body::from(content))
            .unwrap(),
        Err(_) => {
            let body = Body::from(std::fs::read(format!("{root}/404.html")).unwrap_or_default());
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header(header::CONTENT_TYPE, "text/html")
                .body(body)
                .unwrap()
        }
    }
}

#[derive(Clone, Debug)]
pub struct AppState {
    pub state_thing: String,
}
