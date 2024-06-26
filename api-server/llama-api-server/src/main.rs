#[macro_use]
extern crate log;

mod backend;
mod error;
mod utils;

use anyhow::Result;
use chat_prompts::PromptTemplateType;
use clap::Parser;
use error::ServerError;
use hyper::{
    body::HttpBody,
    header,
    server::conn::AddrStream,
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server, StatusCode,
};
use llama_core::MetadataBuilder;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf};
use utils::LogLevel;

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

// server info
pub(crate) static SERVER_INFO: OnceCell<ServerInfo> = OnceCell::new();

// default socket address of LlamaEdge API Server instance
const DEFAULT_SOCKET_ADDRESS: &str = "0.0.0.0:8080";

#[derive(Debug, Parser)]
#[command(name = "LlamaEdge API Server", version = env!("CARGO_PKG_VERSION"), author = env!("CARGO_PKG_AUTHORS"), about = "LlamaEdge API Server")]
struct Cli {
    /// Sets names for chat and/or embedding models. To run both chat and embedding models, the names should be separated by comma without space, for example, '--model-name Llama-2-7b,all-minilm'. The first value is for the chat model, and the second is for the embedding model.
    #[arg(short, long, value_delimiter = ',', default_value = "default")]
    model_name: Vec<String>,
    /// Model aliases for chat and embedding models
    #[arg(
        short = 'a',
        long,
        value_delimiter = ',',
        default_value = "default,embedding"
    )]
    model_alias: Vec<String>,
    /// Sets context sizes for chat and/or embedding models. To run both chat and embedding models, the sizes should be separated by comma without space, for example, '--ctx-size 4096,384'. The first value is for the chat model, and the second is for the embedding model.
    #[arg(
        short = 'c',
        long,
        value_delimiter = ',',
        default_value = "4096,384",
        value_parser = clap::value_parser!(u64)
    )]
    ctx_size: Vec<u64>,
    /// Sets batch sizes for chat and/or embedding models. To run both chat and embedding models, the sizes should be separated by comma without space, for example, '--batch-size 128,64'. The first value is for the chat model, and the second is for the embedding model.
    #[arg(short, long, value_delimiter = ',', default_value = "512,512", value_parser = clap::value_parser!(u64))]
    batch_size: Vec<u64>,
    /// Sets prompt templates for chat and/or embedding models, respectively. To run both chat and embedding models, the prompt templates should be separated by comma without space, for example, '--prompt-template llama-2-chat,embedding'. The first value is for the chat model, and the second is for the embedding model.
    #[arg(short, long, value_delimiter = ',', value_parser = clap::value_parser!(PromptTemplateType), required = true)]
    prompt_template: Vec<PromptTemplateType>,
    /// Halt generation at PROMPT, return control.
    #[arg(short, long)]
    reverse_prompt: Option<String>,
    /// Number of tokens to predict
    #[arg(short, long, default_value = "1024")]
    n_predict: u64,
    /// Number of layers to run on the GPU
    #[arg(short = 'g', long, default_value = "100")]
    n_gpu_layers: u64,
    /// Disable memory mapping for file access of chat models
    #[arg(long)]
    no_mmap: Option<bool>,
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
    /// Socket address of LlamaEdge API Server instance
    #[arg(long, default_value = DEFAULT_SOCKET_ADDRESS)]
    socket_addr: String,
    /// Root path for the Web UI files
    #[arg(long, default_value = "chatbot-ui")]
    web_ui: PathBuf,
    /// Deprecated. Print prompt strings to stdout
    #[arg(long)]
    log_prompts: bool,
    /// Deprecated. Print statistics to stdout
    #[arg(long)]
    log_stat: bool,
    /// Deprecated. Print all log information to stdout
    #[arg(long)]
    log_all: bool,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), ServerError> {
    let mut plugin_debug = false;

    // get the environment variable `LLAMA_LOG`
    let log_level: LogLevel = std::env::var("LLAMA_LOG")
        .unwrap_or("info".to_string())
        .parse()
        .unwrap_or(LogLevel::Info);

    if log_level == LogLevel::Debug || log_level == LogLevel::Trace {
        plugin_debug = true;
    }

    // set global logger
    wasi_logger::Logger::install().expect("failed to install wasi_logger::Logger");
    log::set_max_level(log_level.into());

    // parse the command line arguments
    let cli = Cli::parse();

    // log the version of the server
    info!(target: "server_config", "server version: {}", env!("CARGO_PKG_VERSION"));

    // log model names
    if cli.model_name.is_empty() && cli.model_name.len() > 2 {
        return Err(ServerError::ArgumentError(
            "Invalid setting for model name. For running chat or embedding model, please specify a single model name. For running both chat and embedding models, please specify two model names: the first one for chat model, the other for embedding model.".to_owned(),
        ));
    }
    info!(target: "server_config", "model_name: {}", cli.model_name.join(",").to_string());

    // log model alias
    let mut model_alias = String::new();
    if cli.model_name.len() == 1 {
        model_alias.clone_from(&cli.model_alias[0]);
    } else if cli.model_alias.len() == 2 {
        model_alias = cli.model_alias.join(",").to_string();
    }
    info!(target: "server_config", "model_alias: {}", model_alias);

    // log context size
    if cli.ctx_size.is_empty() && cli.ctx_size.len() > 2 {
        return Err(ServerError::ArgumentError(
            "Invalid setting for context size. For running chat or embedding model, please specify a single context size. For running both chat and embedding models, please specify two context sizes: the first one for chat model, the other for embedding model.".to_owned(),
        ));
    }
    let mut ctx_sizes_str = String::new();
    if cli.model_name.len() == 1 {
        ctx_sizes_str = cli.ctx_size[0].to_string();
    } else if cli.model_name.len() == 2 {
        ctx_sizes_str = cli
            .ctx_size
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<String>>()
            .join(",");
    }
    info!(target: "server_config", "ctx_size: {}", ctx_sizes_str);

    // log batch size
    if cli.batch_size.is_empty() && cli.batch_size.len() > 2 {
        return Err(ServerError::ArgumentError(
            "Invalid setting for batch size. For running chat or embedding model, please specify a single batch size. For running both chat and embedding models, please specify two batch sizes: the first one for chat model, the other for embedding model.".to_owned(),
        ));
    }
    let mut batch_sizes_str = String::new();
    if cli.model_name.len() == 1 {
        batch_sizes_str = cli.batch_size[0].to_string();
    } else if cli.model_name.len() == 2 {
        batch_sizes_str = cli
            .batch_size
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<String>>()
            .join(",");
    }
    info!(target: "server_config", "batch_size: {}", batch_sizes_str);

    // log prompt template
    if cli.prompt_template.is_empty() && cli.prompt_template.len() > 2 {
        return Err(ServerError::ArgumentError(
            "LlamaEdge API server requires prompt templates. For running chat or embedding model, please specify a single prompt template. For running both chat and embedding models, please specify two prompt templates: the first one for chat model, the other for embedding model.".to_owned(),
        ));
    }
    let prompt_template_str: String = cli
        .prompt_template
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<String>>()
        .join(",");
    info!(target: "server_config", "prompt_template: {}", prompt_template_str);
    if cli.model_name.len() != cli.prompt_template.len() {
        return Err(ServerError::ArgumentError(
            "The number of model names and prompt templates must be the same.".to_owned(),
        ));
    }

    // log reverse prompt
    if let Some(reverse_prompt) = &cli.reverse_prompt {
        info!(target: "server_config", "reverse_prompt: {}", reverse_prompt);
    }

    // log n_predict
    info!(target: "server_config", "n_predict: {}", cli.n_predict);

    // log n_gpu_layers
    info!(target: "server_config", "n_gpu_layers: {}", cli.n_gpu_layers);

    // log no_mmap
    if let Some(no_mmap) = &cli.no_mmap {
        info!(
            "[INFO] Disable memory mapping for file access of chat models : {}",
            no_mmap.clone()
        );
    }

    // log temperature
    info!(target: "server_config", "temp: {}", cli.temp);

    // log top-p sampling
    info!(target: "server_config", "top_p: {}", cli.top_p);

    // repeat penalty
    info!(target: "server_config", "repeat_penalty: {}", cli.repeat_penalty);

    // log presence penalty
    info!(target: "server_config", "presence_penalty: {}", cli.presence_penalty);

    // log frequency penalty
    info!(target: "server_config", "frequency_penalty: {}", cli.frequency_penalty);

    // log multimodal projector
    if let Some(llava_mmproj) = &cli.llava_mmproj {
        info!(target: "server_config", "llava_mmproj: {}", llava_mmproj.clone());
    }

    // initialize the core context
    let mut chat_model_config = None;
    let mut embedding_model_config = None;
    if cli.prompt_template.len() == 1 {
        match cli.prompt_template[0] {
            PromptTemplateType::Embedding => {
                // create a Metadata instance
                let metadata_embedding = MetadataBuilder::new(
                    cli.model_name[0].clone(),
                    cli.model_alias[0].clone(),
                    cli.prompt_template[0],
                )
                .with_ctx_size(cli.ctx_size[0])
                .with_batch_size(cli.batch_size[0])
                .enable_plugin_log(true)
                .enable_debug_log(plugin_debug)
                .build();

                // set the embedding model config
                embedding_model_config = Some(ModelConfig {
                    name: metadata_embedding.model_name.clone(),
                    ty: "embedding".to_string(),
                    ctx_size: metadata_embedding.ctx_size,
                    batch_size: metadata_embedding.batch_size,
                    ..Default::default()
                });

                // initialize the core context
                llama_core::init_core_context(None, Some(&[metadata_embedding]))
                    .map_err(|e| ServerError::Operation(format!("{}", e)))?;
            }
            _ => {
                // create a Metadata instance
                let metadata_chat = MetadataBuilder::new(
                    cli.model_name[0].clone(),
                    cli.model_alias[0].clone(),
                    cli.prompt_template[0],
                )
                .with_ctx_size(cli.ctx_size[0])
                .with_batch_size(cli.batch_size[0])
                .with_n_predict(cli.n_predict)
                .with_n_gpu_layers(cli.n_gpu_layers)
                .disable_mmap(cli.no_mmap)
                .with_temperature(cli.temp)
                .with_top_p(cli.top_p)
                .with_repeat_penalty(cli.repeat_penalty)
                .with_presence_penalty(cli.presence_penalty)
                .with_frequency_penalty(cli.frequency_penalty)
                .with_reverse_prompt(cli.reverse_prompt)
                .with_mmproj(cli.llava_mmproj.clone())
                .enable_plugin_log(true)
                .enable_debug_log(plugin_debug)
                .build();

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
                    use_mmap: metadata_chat.use_mmap,
                    temperature: Some(metadata_chat.temperature),
                    top_p: Some(metadata_chat.top_p),
                    repeat_penalty: Some(metadata_chat.repeat_penalty),
                    presence_penalty: Some(metadata_chat.presence_penalty),
                    frequency_penalty: Some(metadata_chat.frequency_penalty),
                });

                // initialize the core context
                llama_core::init_core_context(Some(&[metadata_chat]), None)
                    .map_err(|e| ServerError::Operation(format!("{}", e)))?;
            }
        }
    } else if cli.prompt_template.len() == 2 {
        // create a Metadata instance
        let metadata_chat = MetadataBuilder::new(
            cli.model_name[0].clone(),
            cli.model_alias[0].clone(),
            cli.prompt_template[0],
        )
        .with_ctx_size(cli.ctx_size[0])
        .with_batch_size(cli.batch_size[0])
        .with_n_predict(cli.n_predict)
        .with_n_gpu_layers(cli.n_gpu_layers)
        .disable_mmap(cli.no_mmap)
        .with_temperature(cli.temp)
        .with_top_p(cli.top_p)
        .with_repeat_penalty(cli.repeat_penalty)
        .with_presence_penalty(cli.presence_penalty)
        .with_frequency_penalty(cli.frequency_penalty)
        .with_reverse_prompt(cli.reverse_prompt)
        .with_mmproj(cli.llava_mmproj.clone())
        .enable_plugin_log(true)
        .enable_debug_log(plugin_debug)
        .build();

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
            use_mmap: metadata_chat.use_mmap,
            temperature: Some(metadata_chat.temperature),
            top_p: Some(metadata_chat.top_p),
            repeat_penalty: Some(metadata_chat.repeat_penalty),
            presence_penalty: Some(metadata_chat.presence_penalty),
            frequency_penalty: Some(metadata_chat.frequency_penalty),
        });

        // create a Metadata instance
        let metadata_embedding = MetadataBuilder::new(
            cli.model_name[1].clone(),
            cli.model_alias[1].clone(),
            cli.prompt_template[1],
        )
        .with_ctx_size(cli.ctx_size[1])
        .with_batch_size(cli.batch_size[1])
        .enable_plugin_log(true)
        .enable_debug_log(plugin_debug)
        .build();

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
    }

    // log plugin version
    let plugin_info =
        llama_core::get_plugin_info().map_err(|e| ServerError::Operation(e.to_string()))?;
    let plugin_version = format!(
        "b{build_number} (commit {commit_id})",
        build_number = plugin_info.build_number,
        commit_id = plugin_info.commit_id,
    );
    info!(target: "server_config", "plugin_ggml_version: {}", plugin_version);

    // socket address
    let addr = cli
        .socket_addr
        .parse::<SocketAddr>()
        .map_err(|e| ServerError::SocketAddr(e.to_string()))?;
    let port = addr.port().to_string();

    // log socket address
    info!(target: "server_config", "socket_address: {}", addr.to_string());

    // create server info
    let server_info = ServerInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        plugin_version,
        port,
        chat_model: chat_model_config,
        embedding_model: embedding_model_config,
    };
    SERVER_INFO
        .set(server_info)
        .map_err(|_| ServerError::Operation("Failed to set `SERVER_INFO`.".to_string()))?;

    let new_service = make_service_fn(move |conn: &AddrStream| {
        // log socket address
        info!(target: "connection", "remote_addr: {}, local_addr: {}", conn.remote_addr().to_string(), conn.local_addr().to_string());

        // web ui
        let web_ui = cli.web_ui.to_string_lossy().to_string();

        async move { Ok::<_, Error>(service_fn(move |req| handle_request(req, web_ui.clone()))) }
    });

    let server = Server::bind(&addr).serve(new_service);

    // println!(
    //     "LlamaEdge API server listening on http://{}:{}",
    //     addr.ip(),
    //     addr.port()
    // );

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

    // log request
    {
        let method = hyper::http::Method::as_str(req.method()).to_string();
        let path = req.uri().path().to_string();
        let version = format!("{:?}", req.version());
        if req.method() == hyper::http::Method::POST {
            let size: u64 = req
                .headers()
                .get("content-length")
                .unwrap()
                .to_str()
                .unwrap()
                .parse()
                .unwrap();

            info!(target: "request", "method: {}, endpoint: {}, http_version: {}, size: {}", method, path, version, size);
        } else {
            info!(target: "request", "method: {}, endpoint: {}, http_version: {}", method, path, version);
        }
    }

    let response = match root_path.as_str() {
        "/echo" => Response::new(Body::from("echo test")),
        "/v1" => backend::handle_llama_request(req).await,
        _ => static_response(path_str, web_ui),
    };

    // log response
    {
        let status_code = response.status();
        if status_code.as_u16() < 400 {
            // log response
            let response_version = format!("{:?}", response.version());
            let response_body_size: u64 = response.body().size_hint().lower();
            let response_status = status_code.as_u16();
            let response_is_informational = status_code.is_informational();
            let response_is_success = status_code.is_success();
            let response_is_redirection = status_code.is_redirection();
            let response_is_client_error = status_code.is_client_error();
            let response_is_server_error = status_code.is_server_error();

            info!(target: "response", "version: {}, body_size: {}, status: {}, is_informational: {}, is_success: {}, is_redirection: {}, is_client_error: {}, is_server_error: {}", response_version, response_body_size, response_status, response_is_informational, response_is_success, response_is_redirection, response_is_client_error, response_is_server_error);
        } else {
            let response_version = format!("{:?}", response.version());
            let response_body_size: u64 = response.body().size_hint().lower();
            let response_status = status_code.as_u16();
            let response_is_informational = status_code.is_informational();
            let response_is_success = status_code.is_success();
            let response_is_redirection = status_code.is_redirection();
            let response_is_client_error = status_code.is_client_error();
            let response_is_server_error = status_code.is_server_error();

            error!(target: "response", "version: {}, body_size: {}, status: {}, is_informational: {}, is_success: {}, is_redirection: {}, is_client_error: {}, is_server_error: {}", response_version, response_body_size, response_status, response_is_informational, response_is_success, response_is_redirection, response_is_client_error, response_is_server_error);
        }
    }

    Ok(response)
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
    #[serde(rename = "api_server_version")]
    version: String,
    #[serde(rename = "ggml_plugin_version")]
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
    pub use_mmap: Option<bool>,
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
