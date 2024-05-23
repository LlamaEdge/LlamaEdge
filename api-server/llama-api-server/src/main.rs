mod backend;
mod error;
mod utils;

use anyhow::Result;
use chat_prompts::PromptTemplateType;
use clap::{Args, Parser, Subcommand};
use error::ServerError;
use hyper::{
    header,
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server, StatusCode,
};
use llama_core::MetadataBuilder;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use utils::log;

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

// default socket address of LlamaEdge API Server instance
const DEFAULT_SOCKET_ADDRESS: &str = "0.0.0.0:8080";
// server info
pub(crate) static SERVER_INFO: OnceCell<ServerInfo> = OnceCell::new();

#[derive(Debug, Parser)]
#[command(author, about, version, long_about=None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,

    /// Socket address of LlamaEdge API Server instance
    #[arg(long, default_value = DEFAULT_SOCKET_ADDRESS)]
    socket_addr: String,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Start server for chat completions
    Chat(ChatArgs),
    /// Start server for computing embeddings
    Embedding(EmbeddingArgs),
    /// Start server for both chat completions and computing embeddings
    Max(FullArgs),
}

#[derive(Debug, Args)]
struct ChatArgs {
    /// Name of chat model
    #[arg(long, required = true)]
    model_name: String,
    /// Alias for chat model
    #[arg(long, default_value = "default")]
    model_alias: String,
    /// Context size of chat model
    #[arg(short, long, default_value = "512")]
    ctx_size: u64,
    /// Prompt template used for chat completions
    #[arg(short, long, value_parser = clap::value_parser!(PromptTemplateType), required = true)]
    prompt_template: PromptTemplateType,
    /// Reverse prompt for stopping generation.
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
    /// Root path for the Web UI files
    #[arg(long, default_value = "chatbot-ui")]
    web_ui: PathBuf,
}

#[derive(Debug, Args)]
struct EmbeddingArgs {
    /// Name of embedding model
    #[arg(long, required = true)]
    model_name: String,
    /// Alias for embedding model
    #[arg(long, default_value = "default")]
    model_alias: String,
    /// Context size of embedding model
    #[arg(short, long, default_value = "512")]
    ctx_size: u64,
    /// Batch size for embedding model
    #[arg(short, long, default_value = "512")]
    batch_size: u64,
    /// Print statistics to stdout
    #[arg(long)]
    log_stat: bool,
    /// Print all log information to stdout
    #[arg(long)]
    log_all: bool,
}

#[derive(Debug, Args)]
struct FullArgs {
    // * Chat model arguments
    /// Name of chat model
    #[arg(long, required = true)]
    model_name: String,
    /// Alias for chat model
    #[arg(long, default_value = "default")]
    model_alias: String,
    /// Context size of chat model
    #[arg(long, default_value = "512")]
    ctx_size: u64,
    /// Prompt template used for chat completions
    #[arg(long, value_parser = clap::value_parser!(PromptTemplateType), required = true)]
    prompt_template: PromptTemplateType,
    /// Reverse prompt for stopping generation.
    #[arg(long)]
    reverse_prompt: Option<String>,
    /// Number of tokens to predict
    #[arg(long, default_value = "1024")]
    n_predict: u64,
    /// Number of layers to run on the GPU
    #[arg(long, default_value = "100")]
    n_gpu_layers: u64,
    /// Batch size for prompt processing
    #[arg(long, default_value = "512")]
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

    // * Embedding model arguments
    /// Name of embedding model
    #[arg(long, required = true)]
    embedding_model_name: String,
    /// Alias for embedding model
    #[arg(long, default_value = "embedding")]
    embedding_model_alias: String,
    /// Context size of embedding model
    #[arg(long, default_value = "512")]
    embedding_ctx_size: u64,
    /// Batch size for embedding model
    #[arg(long, default_value = "512")]
    embedding_batch_size: u64,

    /// Print prompt strings to stdout
    #[arg(long)]
    log_prompts: bool,
    /// Print statistics to stdout
    #[arg(long)]
    log_stat: bool,
    /// Print all log information to stdout
    #[arg(long)]
    log_all: bool,
    /// Root path for the Web UI files
    #[arg(long, default_value = "chatbot-ui")]
    web_ui: PathBuf,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), ServerError> {
    // get the environment variable `PLUGIN_DEBUG`
    let plugin_debug = std::env::var("PLUGIN_DEBUG").unwrap_or_default();
    let plugin_debug = match plugin_debug.is_empty() {
        true => false,
        false => plugin_debug.to_lowercase().parse::<bool>().unwrap_or(false),
    };

    // parse the command line arguments
    let cli = Cli::parse();

    // log the version of the server
    let server_version = env!("CARGO_PKG_VERSION").to_string();
    log(format!("[INFO] LlamaEdge-RAG version: {}", &server_version));

    let mut chat_model_config = None;
    let mut embedding_model_config = None;
    let web_ui = match cli.command {
        Commands::Chat(chat_args) => {
            // log
            {
                // log the cli options
                log(format!("[INFO] Model name: {}", &chat_args.model_name));
                log(format!("[INFO] Model alias: {}", &chat_args.model_alias));
                log(format!("[INFO] Context size: {}", &chat_args.ctx_size));
                log(format!("[INFO] Batch size: {}", &chat_args.batch_size));
                log(format!(
                    "[INFO] Prompt template: {}",
                    &chat_args.prompt_template
                ));
                if let Some(reverse_prompt) = &chat_args.reverse_prompt {
                    log(format!("[INFO] reverse prompt: {}", reverse_prompt));
                }
                log(format!(
                    "[INFO] Number of tokens to predict: {}",
                    &chat_args.n_predict
                ));
                log(format!(
                    "[INFO] Number of layers to run on the GPU: {}",
                    &chat_args.n_gpu_layers
                ));
                log(format!(
                    "[INFO] Temperature for sampling: {}",
                    &chat_args.temp
                ));
                log(format!(
                    "[INFO] Top-p sampling (1.0 = disabled): {}",
                    &chat_args.top_p
                ));
                log(format!(
                    "[INFO] Penalize repeat sequence of tokens: {}",
                    &chat_args.repeat_penalty
                ));
                log(format!(
                    "[INFO] Presence penalty (0.0 = disabled): {}",
                    &chat_args.presence_penalty
                ));
                log(format!(
                    "[INFO] Frequency penalty (0.0 = disabled): {}",
                    &chat_args.frequency_penalty
                ));
                if let Some(llava_mmproj) = &chat_args.llava_mmproj {
                    log(format!("[INFO] Multimodal projector: {}", llava_mmproj));
                }
                log(format!(
                    "[INFO] Enable prompt log: {}",
                    &chat_args.log_prompts
                ));
                log(format!("[INFO] Enable plugin log: {}", &chat_args.log_stat));
            }

            // create a Metadata instance
            let metadata = MetadataBuilder::new(
                chat_args.model_name,
                chat_args.model_alias,
                chat_args.prompt_template,
            )
            .with_ctx_size(chat_args.ctx_size)
            .with_batch_size(chat_args.batch_size)
            .with_n_predict(chat_args.n_predict)
            .with_n_gpu_layers(chat_args.n_gpu_layers)
            .with_temperature(chat_args.temp)
            .with_top_p(chat_args.top_p)
            .with_repeat_penalty(chat_args.repeat_penalty)
            .with_presence_penalty(chat_args.presence_penalty)
            .with_frequency_penalty(chat_args.frequency_penalty)
            .with_reverse_prompt(chat_args.reverse_prompt)
            .with_mmproj(chat_args.llava_mmproj.clone())
            .enable_prompts_log(chat_args.log_prompts || chat_args.log_all)
            .enable_plugin_log(chat_args.log_stat || chat_args.log_all || plugin_debug)
            .enable_debug_log(plugin_debug)
            .build();

            // set the chat model config
            chat_model_config = Some(ModelConfig {
                name: metadata.model_name.clone(),
                ty: "chat".to_string(),
                ctx_size: metadata.ctx_size,
                batch_size: metadata.batch_size,
                prompt_template: Some(metadata.prompt_template),
                n_predict: Some(metadata.n_predict),
                reverse_prompt: metadata.reverse_prompt.clone(),
                n_gpu_layers: Some(metadata.n_gpu_layers),
                temperature: Some(metadata.temperature),
                top_p: Some(metadata.top_p),
                repeat_penalty: Some(metadata.repeat_penalty),
                presence_penalty: Some(metadata.presence_penalty),
                frequency_penalty: Some(metadata.frequency_penalty),
            });

            // initialize the core context
            llama_core::init_core_context(Some(&[metadata]), None)
                .map_err(|e| ServerError::Operation(format!("{}", e)))?;

            chat_args.web_ui.to_string_lossy().to_string()
        }
        Commands::Embedding(embedding_args) => {
            // log
            {
                // log the cli options
                log(format!("[INFO] Model name: {}", &embedding_args.model_name));
                log(format!(
                    "[INFO] Model alias: {}",
                    &embedding_args.model_alias
                ));
                log(format!("[INFO] Context size: {}", &embedding_args.ctx_size));
                log(format!("[INFO] Batch size: {}", &embedding_args.batch_size));
            }

            // create metadata for embedding model
            let metadata = MetadataBuilder::new(
                embedding_args.model_name,
                embedding_args.model_alias,
                PromptTemplateType::Llama2Chat, // TODO: dummy value. Embedding model does not use prompt template
            )
            .with_ctx_size(embedding_args.ctx_size)
            .with_batch_size(embedding_args.batch_size)
            .enable_plugin_log(embedding_args.log_stat || embedding_args.log_all || plugin_debug)
            .enable_debug_log(plugin_debug)
            .build();

            // set the embedding model config
            embedding_model_config = Some(ModelConfig {
                name: metadata.model_name.clone(),
                ty: "embedding".to_string(),
                ctx_size: metadata.ctx_size,
                batch_size: metadata.batch_size,
                ..Default::default()
            });

            // initialize the core context
            llama_core::init_core_context(None, Some(&[metadata]))
                .map_err(|e| ServerError::Operation(format!("{}", e)))?;

            String::new()
        }
        Commands::Max(full_args) => {
            // log
            {
                // log the cli options
                log(format!(
                    "[INFO] Name of chat model: {}",
                    &full_args.model_name
                ));
                log(format!(
                    "[INFO] Name of embedding model: {}",
                    &full_args.embedding_model_name
                ));
                log(format!(
                    "[INFO] Alias for chat model: {}",
                    &full_args.model_alias
                ));
                log(format!(
                    "[INFO] Alias for embedding model: {}",
                    &full_args.embedding_model_alias
                ));
                log(format!(
                    "[INFO] Context size of chat model: {}",
                    &full_args.ctx_size
                ));
                log(format!(
                    "[INFO] Context size of embedding model: {}",
                    &full_args.embedding_ctx_size
                ));
                log(format!(
                    "[INFO] Batch size for chat model: {}",
                    &full_args.batch_size
                ));
                log(format!(
                    "[INFO] Batch size for embedding model: {}",
                    &full_args.embedding_batch_size
                ));
                log(format!(
                    "[INFO] Prompt template: {}",
                    &full_args.prompt_template
                ));
                if let Some(reverse_prompt) = &full_args.reverse_prompt {
                    log(format!("[INFO] reverse prompt: {}", reverse_prompt));
                }
                log(format!(
                    "[INFO] Number of tokens to predict: {}",
                    &full_args.n_predict
                ));
                log(format!(
                    "[INFO] Number of layers to run on the GPU: {}",
                    &full_args.n_gpu_layers
                ));
                log(format!(
                    "[INFO] Temperature for sampling: {}",
                    &full_args.temp
                ));
                log(format!(
                    "[INFO] Top-p sampling (1.0 = disabled): {}",
                    &full_args.top_p
                ));
                log(format!(
                    "[INFO] Penalize repeat sequence of tokens: {}",
                    &full_args.repeat_penalty
                ));
                log(format!(
                    "[INFO] Presence penalty (0.0 = disabled): {}",
                    &full_args.presence_penalty
                ));
                log(format!(
                    "[INFO] Frequency penalty (0.0 = disabled): {}",
                    &full_args.frequency_penalty
                ));
                if let Some(llava_mmproj) = &full_args.llava_mmproj {
                    log(format!("[INFO] Multimodal projector: {}", llava_mmproj));
                }
                log(format!(
                    "[INFO] Enable prompt log: {}",
                    &full_args.log_prompts
                ));
                log(format!("[INFO] Enable plugin log: {}", &full_args.log_stat));
            }

            // create metadata for chat model
            let metadata_chat = {
                MetadataBuilder::new(
                    full_args.model_name,
                    full_args.model_alias,
                    full_args.prompt_template,
                )
                .with_ctx_size(full_args.ctx_size)
                .with_batch_size(full_args.batch_size)
                .with_n_predict(full_args.n_predict)
                .with_n_gpu_layers(full_args.n_gpu_layers)
                .with_temperature(full_args.temp)
                .with_top_p(full_args.top_p)
                .with_repeat_penalty(full_args.repeat_penalty)
                .with_presence_penalty(full_args.presence_penalty)
                .with_frequency_penalty(full_args.frequency_penalty)
                .with_reverse_prompt(full_args.reverse_prompt)
                .with_mmproj(full_args.llava_mmproj.clone())
                .enable_prompts_log(full_args.log_prompts || full_args.log_all)
                .enable_plugin_log(full_args.log_stat || full_args.log_all || plugin_debug)
                .enable_debug_log(plugin_debug)
                .build()
            };

            // set the chat model config
            chat_model_config = Some(ModelConfig {
                name: metadata_chat.model_name.clone(),
                ty: "chat".to_string(),
                ctx_size: metadata_chat.ctx_size,
                batch_size: metadata_chat.batch_size,
                prompt_template: Some(metadata_chat.prompt_template),
                n_predict: Some(metadata_chat.n_predict),
                reverse_prompt: metadata_chat.reverse_prompt.clone(),
                n_gpu_layers: Some(metadata_chat.n_gpu_layers),
                temperature: Some(metadata_chat.temperature),
                top_p: Some(metadata_chat.top_p),
                repeat_penalty: Some(metadata_chat.repeat_penalty),
                presence_penalty: Some(metadata_chat.presence_penalty),
                frequency_penalty: Some(metadata_chat.frequency_penalty),
            });

            // create metadata for embedding model
            let metadata_embedding = {
                MetadataBuilder::new(
                    full_args.embedding_model_name,
                    full_args.embedding_model_alias,
                    PromptTemplateType::Llama2Chat, // TODO: dummy value. Embedding model does not use prompt template
                )
                .with_ctx_size(full_args.embedding_ctx_size)
                .with_batch_size(full_args.embedding_batch_size)
                .enable_plugin_log(full_args.log_stat || full_args.log_all || plugin_debug)
                .enable_debug_log(plugin_debug)
                .build()
            };

            // set the embedding model config
            embedding_model_config = Some(ModelConfig {
                name: metadata_embedding.model_name.clone(),
                ty: "embedding".to_string(),
                ctx_size: metadata_embedding.ctx_size,
                batch_size: metadata_embedding.batch_size,
                ..Default::default()
            });

            // initialize the core context
            llama_core::init_core_context(Some(&[metadata_chat]), Some(&[metadata_embedding]))
                .map_err(|e| ServerError::Operation(format!("{}", e)))?;

            full_args.web_ui.to_string_lossy().to_string()
        }
    };
    let ref_web_ui = Arc::new(web_ui);

    // get the plugin version info
    let plugin_info =
        llama_core::get_plugin_info().map_err(|e| ServerError::Operation(e.to_string()))?;
    let plugin_version = format!(
        "b{build_number} (commit {commit_id})",
        build_number = plugin_info.build_number,
        commit_id = plugin_info.commit_id,
    );
    log(format!("[INFO] Wasi-nn-ggml plugin: {}", &plugin_version));

    // socket address
    let addr = cli
        .socket_addr
        .parse::<SocketAddr>()
        .map_err(|e| ServerError::SocketAddr(e.to_string()))?;
    // set the server info
    let port = addr.port().to_string();

    // create server info
    let server_info = ServerInfo {
        version: server_version,
        plugin_version,
        port,
        chat_model: chat_model_config,
        embedding_model: embedding_model_config,
    };
    SERVER_INFO
        .set(server_info)
        .map_err(|_| ServerError::Operation("Failed to set `SERVER_INFO`.".to_string()))?;

    let new_service = make_service_fn(move |_| {
        // let web_ui = cli.web_ui.to_string_lossy().to_string();
        let web_ui = ref_web_ui.clone();

        async move {
            Ok::<_, Error>(service_fn(move |req| {
                handle_request(req, (*web_ui).clone())
            }))
        }
    });

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

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ServerInfo {
    version: String,
    plugin_version: String,
    port: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_model: Option<ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_model: Option<ModelConfig>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct ModelConfig {
    // model name
    name: String,
    // type: chat or embedding
    #[serde(rename = "type")]
    ty: String,
    pub ctx_size: u64,
    pub batch_size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<PromptTemplateType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_predict: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reverse_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
}
