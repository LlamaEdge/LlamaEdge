#[macro_use]
extern crate log;

mod backend;
mod config;
mod error;
mod utils;

use anyhow::Result;
use chat_prompts::PromptTemplateType;
use clap::{ArgGroup, Parser, Subcommand};
use error::ServerError;
use hyper::{
    body::HttpBody,
    header,
    server::conn::AddrStream,
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server, StatusCode,
};
use llama_core::metadata::ggml::GgmlMetadataBuilder;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, path::PathBuf};
use tokio::net::TcpListener;
use utils::LogLevel;

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

// server info
pub(crate) static SERVER_INFO: OnceCell<ServerInfo> = OnceCell::new();

// API key
pub(crate) static LLAMA_API_KEY: OnceCell<String> = OnceCell::new();

// default port
const DEFAULT_PORT: &str = "8080";

#[derive(Debug, Parser)]
#[command(name = "LlamaEdge API Server", version = env!("CARGO_PKG_VERSION"), author = env!("CARGO_PKG_AUTHORS"), about = "LlamaEdge API Server")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    server_args: ServerArgs,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Generate or validate configuration file
    Config {
        /// Path to the configuration file (*.toml)
        #[arg(short, long)]
        file: PathBuf,

        /// Use chat model
        #[arg(short, long, default_value = "false")]
        chat: bool,

        /// Use embedding model
        #[arg(short, long, default_value = "false")]
        embedding: bool,

        /// Use the TTS model
        #[arg(short, long, default_value = "false")]
        tts: bool,
    },
}

#[derive(Debug, Parser)]
#[command(group = ArgGroup::new("socket_address_group").multiple(false).args(&["socket_addr", "port"]))]
struct ServerArgs {
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
    /// Sets logical maximum batch sizes for chat and/or embedding models. To run both chat and embedding models, the sizes should be separated by comma without space, for example, '--batch-size 128,64'. The first value is for the chat model, and the second for the embedding model.
    #[arg(short, long, value_delimiter = ',', default_value = "512,512", value_parser = clap::value_parser!(u64))]
    batch_size: Vec<u64>,
    /// Sets physical maximum batch sizes for chat and/or embedding models. To run both chat and embedding models, the sizes should be separated by comma without space, for example, '--ubatch-size 512,512'. The first value is for the chat model, and the second for the embedding model.
    #[arg(short, long, value_delimiter = ',', default_value = "512,512", value_parser = clap::value_parser!(u64))]
    ubatch_size: Vec<u64>,
    /// Sets prompt templates for chat and/or embedding models, respectively. To run both chat and embedding models, the prompt templates should be separated by comma without space, for example, '--prompt-template llama-2-chat,embedding'. The first value is for the chat model, and the second is for the embedding model.
    #[arg(short, long, value_delimiter = ',', value_parser = clap::value_parser!(PromptTemplateType))]
    prompt_template: Vec<PromptTemplateType>,
    /// Halt generation at PROMPT, return control.
    #[arg(short, long)]
    reverse_prompt: Option<String>,
    /// Number of tokens to predict, -1 = infinity, -2 = until context filled.
    #[arg(short, long, default_value = "-1")]
    n_predict: i32,
    /// Number of layers to run on the GPU
    #[arg(short = 'g', long, default_value = "100")]
    n_gpu_layers: u64,
    /// Split the model across multiple GPUs. Possible values: `none` (use one GPU only), `layer` (split layers and KV across GPUs, default), `row` (split rows across GPUs)
    #[arg(long, default_value = "layer")]
    split_mode: String,
    /// The main GPU to use.
    #[arg(long)]
    main_gpu: Option<u64>,
    /// How split tensors should be distributed accross GPUs. If None the model is not split; otherwise, a comma-separated list of non-negative values, e.g., "3,2" presents 60% of the data to GPU 0 and 40% to GPU 1.
    #[arg(long)]
    tensor_split: Option<String>,
    /// Number of threads to use during computation
    #[arg(long, default_value = "2")]
    threads: u64,
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
    /// BNF-like grammar to constrain generations (see samples in grammars/ dir).
    #[arg(long, default_value = "")]
    grammar: String,
    /// JSON schema to constrain generations (<https://json-schema.org/>), e.g. `{}` for any JSON object. For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead.
    #[arg(long)]
    json_schema: Option<String>,
    /// Path to the multimodal projector file
    #[arg(long)]
    llava_mmproj: Option<String>,
    /// Whether to include usage in the stream response. Defaults to false.
    #[arg(long, default_value = "false")]
    include_usage: bool,
    /// Socket address of LlamaEdge API Server instance. For example, `0.0.0.0:8080`.
    #[arg(long, default_value = None, value_parser = clap::value_parser!(SocketAddr), group = "socket_address_group")]
    socket_addr: Option<SocketAddr>,
    /// Port number
    #[arg(long, default_value = DEFAULT_PORT, value_parser = clap::value_parser!(u16), group = "socket_address_group")]
    port: u16,
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

#[allow(clippy::needless_return)]
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

    if let Ok(api_key) = std::env::var("API_KEY") {
        // define a const variable for the API key
        if let Err(e) = LLAMA_API_KEY.set(api_key) {
            let err_msg = format!("Failed to set API key. {}", e);

            error!(target: "stdout", "{}", err_msg);

            return Err(ServerError::Operation(err_msg));
        }
    }

    info!(target: "stdout", "LOG LEVEL: {}", log_level);

    // log the version of the server
    info!(target: "stdout", "SERVER VERSION: {}", env!("CARGO_PKG_VERSION"));

    // parse the command line arguments
    let cli = Cli::parse();

    // Handle subcommands
    if let Some(command) = cli.command {
        match command {
            Commands::Config {
                file,
                chat,
                embedding,
                tts,
            } => {
                if !chat && !embedding && !tts {
                    let err_msg = "Specify at least one of the following: chat, embedding, and/or TTS. by using --chat, --embedding, and/or --tts.";

                    error!(target: "stdout", "{}", err_msg);

                    return Err(ServerError::Operation(err_msg.to_string()));
                }

                info!(target: "stdout", "CONFIG FILE: {}", file.to_string_lossy().to_string());
                let config = config::Config::load(&file)?;

                // chat model
                let mut chat_model_config = None;
                let mut metadata_for_chats = None;
                if chat {
                    info!(target: "stdout", "chat model name: {}", config.chat.model_name);

                    info!(target: "stdout", "chat model alias: {}", config.chat.model_alias);

                    info!(target: "stdout", "chat context size: {}", config.chat.ctx_size);

                    info!(target: "stdout", "chat batch size: {}", config.chat.batch_size);

                    info!(target: "stdout", "chat ubatch size: {}", config.chat.ubatch_size);

                    info!(target: "stdout", "chat prompt template: {}", config.chat.prompt_template);

                    info!(target: "stdout", "chat split mode: {}", config.chat.split_mode);

                    info!(target: "stdout", "chat main gpu: {:?}", config.chat.main_gpu);

                    info!(target: "stdout", "chat tensor split: {:?}", config.chat.tensor_split);

                    info!(target: "stdout", "chat threads: {}", config.chat.threads);

                    info!(target: "stdout", "chat no_mmap: {}", config.chat.no_mmap);

                    info!(target: "stdout", "chat temp: {}", config.chat.temp);

                    info!(target: "stdout", "chat top_p: {}", config.chat.top_p);

                    info!(target: "stdout", "chat repeat_penalty: {}", config.chat.repeat_penalty);

                    info!(target: "stdout", "chat presence_penalty: {}", config.chat.presence_penalty);

                    info!(target: "stdout", "chat frequency_penalty: {}", config.chat.frequency_penalty);

                    info!(target: "stdout", "chat grammar: {:?}", config.chat.grammar);

                    info!(target: "stdout", "chat json_schema: {:?}", config.chat.json_schema);

                    info!(target: "stdout", "chat llava_mmproj: {:?}", config.chat.llava_mmproj);

                    info!(target: "stdout", "chat include_usage: {}", config.chat.include_usage);

                    // create a Metadata instance
                    let metadata_chat = GgmlMetadataBuilder::new(
                        config.chat.model_name,
                        config.chat.model_alias,
                        config.chat.prompt_template,
                    )
                    .with_ctx_size(config.chat.ctx_size)
                    .with_batch_size(config.chat.batch_size)
                    .with_ubatch_size(config.chat.ubatch_size)
                    .with_n_predict(config.chat.n_predict)
                    .with_n_gpu_layers(config.chat.n_gpu_layers)
                    .with_split_mode(config.chat.split_mode)
                    .with_main_gpu(config.chat.main_gpu)
                    .with_tensor_split(config.chat.tensor_split)
                    .with_threads(config.chat.threads)
                    .disable_mmap(Some(config.chat.no_mmap))
                    .with_temperature(config.chat.temp)
                    .with_top_p(config.chat.top_p)
                    .with_repeat_penalty(config.chat.repeat_penalty)
                    .with_presence_penalty(config.chat.presence_penalty)
                    .with_frequency_penalty(config.chat.frequency_penalty)
                    .with_grammar(config.chat.grammar.unwrap_or_default())
                    .with_json_schema(config.chat.json_schema)
                    .with_reverse_prompt(config.chat.reverse_prompt)
                    .with_mmproj(
                        config
                            .chat
                            .llava_mmproj
                            .map(|p| p.to_string_lossy().to_string()),
                    )
                    .enable_plugin_log(true)
                    .enable_debug_log(plugin_debug)
                    .include_usage(config.chat.include_usage)
                    .build();

                    // set the chat model config
                    chat_model_config = Some(ModelConfig {
                        name: metadata_chat.model_name.clone(),
                        ty: "chat".to_string(),
                        ctx_size: metadata_chat.ctx_size,
                        batch_size: metadata_chat.batch_size,
                        ubatch_size: metadata_chat.ubatch_size,
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
                        split_mode: Some(metadata_chat.split_mode.clone()),
                        main_gpu: metadata_chat.main_gpu,
                        tensor_split: metadata_chat.tensor_split.clone(),
                    });

                    metadata_for_chats = Some(vec![metadata_chat]);
                }

                // embedding model
                let mut embedding_model_config = None;
                let mut metadata_for_embeddings = None;
                if embedding {
                    info!(target: "stdout", "embedding model name: {}", config.embedding.model_name);

                    info!(target: "stdout", "embedding model alias: {}", config.embedding.model_alias);

                    info!(target: "stdout", "embedding context size: {}", config.embedding.ctx_size);

                    info!(target: "stdout", "embedding batch size: {}", config.embedding.batch_size);

                    info!(target: "stdout", "embedding ubatch size: {}", config.embedding.ubatch_size);

                    info!(target: "stdout", "embedding split mode: {}", config.embedding.split_mode);

                    info!(target: "stdout", "embedding main gpu: {:?}", config.embedding.main_gpu);

                    info!(target: "stdout", "embedding tensor split: {:?}", config.embedding.tensor_split);

                    info!(target: "stdout", "embedding threads: {}", config.embedding.threads);

                    // create a Metadata instance
                    let metadata_embedding = GgmlMetadataBuilder::new(
                        config.embedding.model_name,
                        config.embedding.model_alias,
                        PromptTemplateType::Embedding,
                    )
                    .with_ctx_size(config.embedding.ctx_size)
                    .with_batch_size(config.embedding.batch_size)
                    .with_ubatch_size(config.embedding.ubatch_size)
                    .with_split_mode(config.embedding.split_mode)
                    .with_main_gpu(config.embedding.main_gpu)
                    .with_tensor_split(config.embedding.tensor_split)
                    .with_threads(config.embedding.threads)
                    .enable_plugin_log(true)
                    .enable_debug_log(plugin_debug)
                    .build();

                    // set the embedding model config
                    embedding_model_config = Some(ModelConfig {
                        name: metadata_embedding.model_name.clone(),
                        ty: "embedding".to_string(),
                        ctx_size: metadata_embedding.ctx_size,
                        batch_size: metadata_embedding.batch_size,
                        ubatch_size: metadata_embedding.ubatch_size,
                        prompt_template: Some(PromptTemplateType::Embedding),
                        n_predict: Some(cli.server_args.n_predict),
                        reverse_prompt: metadata_embedding.reverse_prompt.clone(),
                        n_gpu_layers: Some(metadata_embedding.n_gpu_layers),
                        use_mmap: metadata_embedding.use_mmap,
                        temperature: Some(metadata_embedding.temperature),
                        top_p: Some(metadata_embedding.top_p),
                        repeat_penalty: Some(metadata_embedding.repeat_penalty),
                        presence_penalty: Some(metadata_embedding.presence_penalty),
                        frequency_penalty: Some(metadata_embedding.frequency_penalty),
                        split_mode: Some(metadata_embedding.split_mode.clone()),
                        main_gpu: metadata_embedding.main_gpu,
                        tensor_split: metadata_embedding.tensor_split.clone(),
                    });

                    metadata_for_embeddings = Some(vec![metadata_embedding]);
                }

                // tts model
                let mut tts_model_config = None;
                let mut metadata_for_tts = None;
                if tts {
                    info!(target: "stdout", "tts model name: {}", config.tts.model_name);

                    info!(target: "stdout", "tts model alias: {}", config.tts.model_alias);

                    info!(target: "stdout", "tts codec model: {:?}", config.tts.codec_model);

                    info!(target: "stdout", "tts output file: {}", config.tts.output_file);

                    info!(target: "stdout", "tts context size: {}", config.tts.ctx_size);

                    info!(target: "stdout", "tts batch size: {}", config.tts.batch_size);

                    info!(target: "stdout", "tts ubatch size: {}", config.tts.ubatch_size);

                    info!(target: "stdout", "tts n_predict: {}", config.tts.n_predict);

                    info!(target: "stdout", "tts n_gpu_layers: {}", config.tts.n_gpu_layers);

                    // create a Metadata instance
                    let metadata_tts = GgmlMetadataBuilder::new(
                        config.tts.model_name,
                        config.tts.model_alias,
                        PromptTemplateType::Tts,
                    )
                    .with_ctx_size(config.tts.ctx_size)
                    .with_batch_size(config.tts.batch_size)
                    .with_ubatch_size(config.tts.ubatch_size)
                    .enable_plugin_log(true)
                    .enable_debug_log(plugin_debug)
                    .build();

                    // set the tts model config
                    tts_model_config = Some(ModelConfig {
                        name: metadata_tts.model_name.clone(),
                        ty: "tts".to_string(),
                        ctx_size: metadata_tts.ctx_size,
                        batch_size: metadata_tts.batch_size,
                        ubatch_size: metadata_tts.ubatch_size,
                        prompt_template: Some(PromptTemplateType::Tts),
                        n_predict: Some(metadata_tts.n_predict),
                        reverse_prompt: None,
                        n_gpu_layers: None,
                        use_mmap: None,
                        temperature: None,
                        top_p: None,
                        repeat_penalty: None,
                        presence_penalty: None,
                        frequency_penalty: None,
                        split_mode: None,
                        main_gpu: None,
                        tensor_split: None,
                    });

                    metadata_for_tts = Some(vec![metadata_tts]);
                }

                if metadata_for_chats.is_none() && metadata_for_embeddings.is_none() {
                    let err_msg = "No chat, embedding, and/or TTS configuration is specified.";

                    error!(target: "stdout", "{}", err_msg);

                    return Err(ServerError::Operation(err_msg.to_string()));
                }

                // initialize the core context
                llama_core::init_ggml_context(
                    metadata_for_chats.as_deref(),
                    metadata_for_embeddings.as_deref(),
                    metadata_for_tts.as_deref(),
                )
                .map_err(|e| ServerError::Operation(format!("{}", e)))?;

                // log plugin version
                let plugin_info = llama_core::get_plugin_info()
                    .map_err(|e| ServerError::Operation(e.to_string()))?;
                let plugin_version = format!(
                    "b{build_number} (commit {commit_id})",
                    build_number = plugin_info.build_number,
                    commit_id = plugin_info.commit_id,
                );
                info!(target: "stdout", "plugin_ggml_version: {}", plugin_version);

                // socket address
                let addr = config.server.socket_addr;
                let port = addr.port().to_string();

                // get the environment variable `NODE_VERSION`
                // Note that this is for satisfying the requirement of `gaianet-node` project.
                let node = std::env::var("NODE_VERSION").ok();
                if node.is_some() {
                    // log node version
                    info!(target: "stdout", "gaianet_node_version: {}", node.as_ref().unwrap());
                }

                // create server info
                let server_info = ServerInfo {
                    node,
                    server: ApiServer {
                        ty: "llama".to_string(),
                        version: env!("CARGO_PKG_VERSION").to_string(),
                        plugin_version,
                        port,
                    },
                    chat_model: chat_model_config,
                    embedding_model: embedding_model_config,
                    tts_model: tts_model_config,
                    extras: HashMap::new(),
                };
                SERVER_INFO.set(server_info).map_err(|_| {
                    ServerError::Operation("Failed to set `SERVER_INFO`.".to_string())
                })?;

                let new_service = make_service_fn(move |conn: &AddrStream| {
                    // log socket address
                    info!(target: "stdout", "remote_addr: {}, local_addr: {}", conn.remote_addr().to_string(), conn.local_addr().to_string());

                    // web ui
                    let web_ui = cli.server_args.web_ui.to_string_lossy().to_string();

                    async move {
                        Ok::<_, Error>(service_fn(move |req| handle_request(req, web_ui.clone())))
                    }
                });

                let tcp_listener = TcpListener::bind(addr).await.unwrap();
                info!(target: "stdout", "Listening on {}", addr);

                let server = Server::from_tcp(tcp_listener.into_std().unwrap())
                    .unwrap()
                    .serve(new_service);

                match server.await {
                    Ok(_) => Ok(()),
                    Err(e) => Err(ServerError::Operation(e.to_string())),
                }
            }
        }
    } else {
        // log model names
        if cli.server_args.model_name.is_empty() && cli.server_args.model_name.len() > 2 {
            return Err(ServerError::ArgumentError(
            "Invalid setting for model name. For running chat or embedding model, please specify a single model name. For running both chat and embedding models, please specify two model names: the first one for chat model, the other for embedding model.".to_owned(),
        ));
        }
        info!(target: "stdout", "model_name: {}", cli.server_args.model_name.join(",").to_string());

        // log model alias
        let mut model_alias = String::new();
        if cli.server_args.model_name.len() == 1 {
            model_alias.clone_from(&cli.server_args.model_alias[0]);
        } else if cli.server_args.model_alias.len() == 2 {
            model_alias = cli.server_args.model_alias.join(",").to_string();
        }
        info!(target: "stdout", "model_alias: {}", model_alias);

        // log context size
        if cli.server_args.ctx_size.is_empty() && cli.server_args.ctx_size.len() > 2 {
            return Err(ServerError::ArgumentError(
            "Invalid setting for context size. For running chat or embedding model, please specify a single context size. For running both chat and embedding models, please specify two context sizes: the first one for chat model, the other for embedding model.".to_owned(),
        ));
        }
        let mut ctx_sizes_str = String::new();
        if cli.server_args.model_name.len() == 1 {
            ctx_sizes_str = cli.server_args.ctx_size[0].to_string();
        } else if cli.server_args.model_name.len() == 2 {
            ctx_sizes_str = cli
                .server_args
                .ctx_size
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<String>>()
                .join(",");
        }
        info!(target: "stdout", "ctx_size: {}", ctx_sizes_str);

        // log batch size
        if cli.server_args.batch_size.is_empty() && cli.server_args.batch_size.len() > 2 {
            return Err(ServerError::ArgumentError(
            "Invalid setting for batch size. For running chat or embedding model, please specify a single batch size. For running both chat and embedding models, please specify two batch sizes: the first one for chat model, the other for embedding model.".to_owned(),
        ));
        }
        let mut batch_sizes_str = String::new();
        if cli.server_args.model_name.len() == 1 {
            batch_sizes_str = cli.server_args.batch_size[0].to_string();
        } else if cli.server_args.model_name.len() == 2 {
            batch_sizes_str = cli
                .server_args
                .batch_size
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<String>>()
                .join(",");
        }
        info!(target: "stdout", "batch_size: {}", batch_sizes_str);

        // log ubatch size
        let mut ubatch_sizes_str = String::new();
        if cli.server_args.model_name.len() == 1 {
            ubatch_sizes_str = cli.server_args.ubatch_size[0].to_string();
        } else if cli.server_args.model_name.len() == 2 {
            ubatch_sizes_str = cli
                .server_args
                .ubatch_size
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<String>>()
                .join(",");
        }
        info!(target: "stdout", "ubatch_size: {}", ubatch_sizes_str);

        // log prompt template
        if cli.server_args.prompt_template.is_empty() && cli.server_args.prompt_template.len() > 2 {
            return Err(ServerError::ArgumentError(
                "LlamaEdge API server requires prompt templates. For running chat or embedding model, please specify a single prompt template. For running both chat and embedding models, please specify two prompt templates: the first one for chat model, the other for embedding model.".to_owned(),
            ));
        }

        let prompt_template_str: String = cli
            .server_args
            .prompt_template
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<String>>()
            .join(",");
        info!(target: "stdout", "prompt_template: {}", prompt_template_str);
        if cli.server_args.model_name.len() != cli.server_args.prompt_template.len() {
            return Err(ServerError::ArgumentError(
                "The number of model names and prompt templates must be the same.".to_owned(),
            ));
        }

        // log reverse prompt
        if let Some(reverse_prompt) = &cli.server_args.reverse_prompt {
            info!(target: "stdout", "reverse_prompt: {}", reverse_prompt);
        }

        // log n_predict
        info!(target: "stdout", "n_predict: {}", cli.server_args.n_predict);

        // log n_gpu_layers
        info!(target: "stdout", "n_gpu_layers: {}", cli.server_args.n_gpu_layers);

        // log split_mode
        info!(target: "stdout", "split_mode: {}", cli.server_args.split_mode);

        // log main_gpu
        if let Some(main_gpu) = &cli.server_args.main_gpu {
            info!(target: "stdout", "main_gpu: {}", main_gpu);
        }

        // log tensor_split
        if let Some(tensor_split) = &cli.server_args.tensor_split {
            info!(target: "stdout", "tensor_split: {}", tensor_split);
        }

        // log threads
        info!(target: "stdout", "threads: {}", cli.server_args.threads);

        // log no_mmap
        if let Some(no_mmap) = &cli.server_args.no_mmap {
            info!(target: "stdout", "no_mmap: {}", no_mmap);
        }

        // log temperature
        info!(target: "stdout", "temp: {}", cli.server_args.temp);

        // log top-p sampling
        info!(target: "stdout", "top_p: {}", cli.server_args.top_p);

        // repeat penalty
        info!(target: "stdout", "repeat_penalty: {}", cli.server_args.repeat_penalty);

        // log presence penalty
        info!(target: "stdout", "presence_penalty: {}", cli.server_args.presence_penalty);

        // log frequency penalty
        info!(target: "stdout", "frequency_penalty: {}", cli.server_args.frequency_penalty);

        // log grammar
        if !cli.server_args.grammar.is_empty() {
            info!(target: "stdout", "grammar: {}", &cli.server_args.grammar);
        }

        // log json schema
        if let Some(json_schema) = &cli.server_args.json_schema {
            info!(target: "stdout", "json_schema: {}", json_schema);
        }

        // log multimodal projector
        if let Some(llava_mmproj) = &cli.server_args.llava_mmproj {
            info!(target: "stdout", "llava_mmproj: {}", llava_mmproj);
        }

        // log include_usage
        info!(target: "stdout", "include_usage: {}", cli.server_args.include_usage);

        // initialize the core context
        let mut chat_model_config = None;
        let mut embedding_model_config = None;
        if cli.server_args.prompt_template.len() == 1 {
            match cli.server_args.prompt_template[0] {
                PromptTemplateType::Embedding => {
                    // create a Metadata instance
                    let metadata_embedding = GgmlMetadataBuilder::new(
                        cli.server_args.model_name[0].clone(),
                        cli.server_args.model_alias[0].clone(),
                        cli.server_args.prompt_template[0],
                    )
                    .with_ctx_size(cli.server_args.ctx_size[0])
                    .with_batch_size(cli.server_args.batch_size[0])
                    .with_ubatch_size(cli.server_args.ubatch_size[0])
                    .with_split_mode(cli.server_args.split_mode)
                    .with_main_gpu(cli.server_args.main_gpu)
                    .with_tensor_split(cli.server_args.tensor_split)
                    .with_threads(cli.server_args.threads)
                    .enable_plugin_log(true)
                    .enable_debug_log(plugin_debug)
                    .build();

                    // set the embedding model config
                    embedding_model_config = Some(ModelConfig {
                        name: metadata_embedding.model_name.clone(),
                        ty: "embedding".to_string(),
                        ctx_size: metadata_embedding.ctx_size,
                        batch_size: metadata_embedding.batch_size,
                        ubatch_size: metadata_embedding.ubatch_size,
                        prompt_template: Some(PromptTemplateType::Embedding),
                        n_predict: Some(cli.server_args.n_predict),
                        reverse_prompt: metadata_embedding.reverse_prompt.clone(),
                        n_gpu_layers: Some(metadata_embedding.n_gpu_layers),
                        use_mmap: metadata_embedding.use_mmap,
                        temperature: Some(metadata_embedding.temperature),
                        top_p: Some(metadata_embedding.top_p),
                        repeat_penalty: Some(metadata_embedding.repeat_penalty),
                        presence_penalty: Some(metadata_embedding.presence_penalty),
                        frequency_penalty: Some(metadata_embedding.frequency_penalty),
                        split_mode: Some(metadata_embedding.split_mode.clone()),
                        main_gpu: metadata_embedding.main_gpu,
                        tensor_split: metadata_embedding.tensor_split.clone(),
                    });

                    // initialize the core context
                    llama_core::init_ggml_context(None, Some(&[metadata_embedding]), None)
                        .map_err(|e| ServerError::Operation(format!("{}", e)))?;
                }
                _ => {
                    // create a Metadata instance
                    let metadata_chat = GgmlMetadataBuilder::new(
                        cli.server_args.model_name[0].clone(),
                        cli.server_args.model_alias[0].clone(),
                        cli.server_args.prompt_template[0],
                    )
                    .with_ctx_size(cli.server_args.ctx_size[0])
                    .with_batch_size(cli.server_args.batch_size[0])
                    .with_ubatch_size(cli.server_args.ubatch_size[0])
                    .with_n_predict(cli.server_args.n_predict)
                    .with_n_gpu_layers(cli.server_args.n_gpu_layers)
                    .with_split_mode(cli.server_args.split_mode)
                    .with_main_gpu(cli.server_args.main_gpu)
                    .with_tensor_split(cli.server_args.tensor_split)
                    .with_threads(cli.server_args.threads)
                    .disable_mmap(cli.server_args.no_mmap)
                    .with_temperature(cli.server_args.temp)
                    .with_top_p(cli.server_args.top_p)
                    .with_repeat_penalty(cli.server_args.repeat_penalty)
                    .with_presence_penalty(cli.server_args.presence_penalty)
                    .with_frequency_penalty(cli.server_args.frequency_penalty)
                    .with_grammar(cli.server_args.grammar)
                    .with_json_schema(cli.server_args.json_schema)
                    .with_reverse_prompt(cli.server_args.reverse_prompt)
                    .with_mmproj(cli.server_args.llava_mmproj.clone())
                    .enable_plugin_log(true)
                    .enable_debug_log(plugin_debug)
                    .include_usage(cli.server_args.include_usage)
                    .build();

                    // set the chat model config
                    chat_model_config = Some(ModelConfig {
                        name: metadata_chat.model_name.clone(),
                        ty: "chat".to_string(),
                        ctx_size: metadata_chat.ctx_size,
                        batch_size: metadata_chat.batch_size,
                        ubatch_size: metadata_chat.ubatch_size,
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
                        split_mode: Some(metadata_chat.split_mode.clone()),
                        main_gpu: metadata_chat.main_gpu,
                        tensor_split: metadata_chat.tensor_split.clone(),
                    });

                    // initialize the core context
                    llama_core::init_ggml_context(Some(&[metadata_chat]), None, None)
                        .map_err(|e| ServerError::Operation(format!("{}", e)))?;
                }
            }
        } else if cli.server_args.prompt_template.len() == 2 {
            // create a Metadata instance
            let metadata_chat = GgmlMetadataBuilder::new(
                cli.server_args.model_name[0].clone(),
                cli.server_args.model_alias[0].clone(),
                cli.server_args.prompt_template[0],
            )
            .with_ctx_size(cli.server_args.ctx_size[0])
            .with_batch_size(cli.server_args.batch_size[0])
            .with_ubatch_size(cli.server_args.ubatch_size[0])
            .with_n_predict(cli.server_args.n_predict)
            .with_n_gpu_layers(cli.server_args.n_gpu_layers)
            .with_split_mode(cli.server_args.split_mode.clone())
            .with_main_gpu(cli.server_args.main_gpu)
            .with_tensor_split(cli.server_args.tensor_split.clone())
            .with_threads(cli.server_args.threads)
            .disable_mmap(cli.server_args.no_mmap)
            .with_temperature(cli.server_args.temp)
            .with_top_p(cli.server_args.top_p)
            .with_repeat_penalty(cli.server_args.repeat_penalty)
            .with_presence_penalty(cli.server_args.presence_penalty)
            .with_frequency_penalty(cli.server_args.frequency_penalty)
            .with_grammar(cli.server_args.grammar)
            .with_json_schema(cli.server_args.json_schema)
            .with_reverse_prompt(cli.server_args.reverse_prompt)
            .with_mmproj(cli.server_args.llava_mmproj.clone())
            .enable_plugin_log(true)
            .enable_debug_log(plugin_debug)
            .include_usage(cli.server_args.include_usage)
            .build();

            // set the chat model config
            chat_model_config = Some(ModelConfig {
                name: metadata_chat.model_name.clone(),
                ty: "chat".to_string(),
                ctx_size: metadata_chat.ctx_size,
                batch_size: metadata_chat.batch_size,
                ubatch_size: metadata_chat.ubatch_size,
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
                split_mode: Some(metadata_chat.split_mode.clone()),
                main_gpu: metadata_chat.main_gpu,
                tensor_split: metadata_chat.tensor_split.clone(),
            });

            // create a Metadata instance
            let metadata_embedding = GgmlMetadataBuilder::new(
                cli.server_args.model_name[1].clone(),
                cli.server_args.model_alias[1].clone(),
                cli.server_args.prompt_template[1],
            )
            .with_ctx_size(cli.server_args.ctx_size[1])
            .with_batch_size(cli.server_args.batch_size[1])
            .with_ubatch_size(cli.server_args.ubatch_size[1])
            .with_split_mode(cli.server_args.split_mode)
            .with_main_gpu(cli.server_args.main_gpu)
            .with_tensor_split(cli.server_args.tensor_split)
            .with_threads(cli.server_args.threads)
            .enable_plugin_log(true)
            .enable_debug_log(plugin_debug)
            .build();

            // set the embedding model config
            embedding_model_config = Some(ModelConfig {
                name: metadata_embedding.model_name.clone(),
                ty: "embedding".to_string(),
                ctx_size: metadata_embedding.ctx_size,
                batch_size: metadata_embedding.batch_size,
                ubatch_size: metadata_embedding.ubatch_size,
                prompt_template: Some(PromptTemplateType::Embedding),
                n_predict: Some(cli.server_args.n_predict),
                reverse_prompt: metadata_embedding.reverse_prompt.clone(),
                n_gpu_layers: Some(metadata_embedding.n_gpu_layers),
                use_mmap: metadata_embedding.use_mmap,
                temperature: Some(metadata_embedding.temperature),
                top_p: Some(metadata_embedding.top_p),
                repeat_penalty: Some(metadata_embedding.repeat_penalty),
                presence_penalty: Some(metadata_embedding.presence_penalty),
                frequency_penalty: Some(metadata_embedding.frequency_penalty),
                split_mode: Some(metadata_embedding.split_mode.clone()),
                main_gpu: metadata_embedding.main_gpu,
                tensor_split: metadata_embedding.tensor_split.clone(),
            });

            // initialize the core context
            llama_core::init_ggml_context(
                Some(&[metadata_chat]),
                Some(&[metadata_embedding]),
                None,
            )
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
        info!(target: "stdout", "plugin_ggml_version: {}", plugin_version);

        // socket address
        let addr = match cli.server_args.socket_addr {
            Some(addr) => addr,
            None => SocketAddr::from(([0, 0, 0, 0], cli.server_args.port)),
        };
        let port = addr.port().to_string();

        // get the environment variable `NODE_VERSION`
        // Note that this is for satisfying the requirement of `gaianet-node` project.
        let node = std::env::var("NODE_VERSION").ok();
        if node.is_some() {
            // log node version
            info!(target: "stdout", "gaianet_node_version: {}", node.as_ref().unwrap());
        }

        // create server info
        let server_info = ServerInfo {
            node,
            server: ApiServer {
                ty: "llama".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                plugin_version,
                port,
            },
            chat_model: chat_model_config,
            embedding_model: embedding_model_config,
            tts_model: None,
            extras: HashMap::new(),
        };
        SERVER_INFO
            .set(server_info)
            .map_err(|_| ServerError::Operation("Failed to set `SERVER_INFO`.".to_string()))?;

        let new_service = make_service_fn(move |conn: &AddrStream| {
            // log socket address
            info!(target: "stdout", "remote_addr: {}, local_addr: {}", conn.remote_addr().to_string(), conn.local_addr().to_string());

            // web ui
            let web_ui = cli.server_args.web_ui.to_string_lossy().to_string();

            async move { Ok::<_, Error>(service_fn(move |req| handle_request(req, web_ui.clone()))) }
        });

        let tcp_listener = TcpListener::bind(addr).await.unwrap();
        info!(target: "stdout", "Listening on {}", addr);

        let server = Server::from_tcp(tcp_listener.into_std().unwrap())
            .unwrap()
            .serve(new_service);

        match server.await {
            Ok(_) => Ok(()),
            Err(e) => Err(ServerError::Operation(e.to_string())),
        }
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

    // check if the API key is valid
    if let Some(auth_header) = req.headers().get("authorization") {
        if !auth_header.is_empty() {
            let auth_header = match auth_header.to_str() {
                Ok(auth_header) => auth_header,
                Err(e) => {
                    let err_msg = format!("Failed to get authorization header: {}", e);
                    return Ok(error::unauthorized(err_msg));
                }
            };

            let api_key = auth_header.split(" ").nth(1).unwrap_or_default();
            info!(target: "stdout", "API Key: {}", api_key);

            if let Some(stored_api_key) = LLAMA_API_KEY.get() {
                if api_key != stored_api_key {
                    let err_msg = "Invalid API key.";
                    return Ok(error::unauthorized(err_msg));
                }
            }
        }
    }

    // log request
    {
        let method = hyper::http::Method::as_str(req.method()).to_string();
        let path = req.uri().path().to_string();
        let version = format!("{:?}", req.version());

        if req.method() == hyper::http::Method::POST {
            let size: u64 = match req.headers().get("content-length") {
                Some(content_length) => content_length.to_str().unwrap().parse().unwrap(),
                None => 0,
            };

            info!(target: "stdout", "method: {}, http_version: {}, content-length: {}", method, version, size);
            info!(target: "stdout", "endpoint: {}", path);
        } else {
            info!(target: "stdout", "method: {}, http_version: {}", method, version);
            info!(target: "stdout", "endpoint: {}", path);
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
            info!(target: "stdout", "response_version: {}", response_version);
            let response_body_size: u64 = response.body().size_hint().lower();
            info!(target: "stdout", "response_body_size: {}", response_body_size);
            let response_status = status_code.as_u16();
            info!(target: "stdout", "response_status: {}", response_status);
            let response_is_success = status_code.is_success();
            info!(target: "stdout", "response_is_success: {}", response_is_success);
        } else {
            let response_version = format!("{:?}", response.version());
            error!(target: "stdout", "response_version: {}", response_version);
            let response_body_size: u64 = response.body().size_hint().lower();
            error!(target: "stdout", "response_body_size: {}", response_body_size);
            let response_status = status_code.as_u16();
            error!(target: "stdout", "response_status: {}", response_status);
            let response_is_success = status_code.is_success();
            error!(target: "stdout", "response_is_success: {}", response_is_success);
            let response_is_client_error = status_code.is_client_error();
            error!(target: "stdout", "response_is_client_error: {}", response_is_client_error);
            let response_is_server_error = status_code.is_server_error();
            error!(target: "stdout", "response_is_server_error: {}", response_is_server_error);
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
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "node_version")]
    node: Option<String>,
    #[serde(rename = "api_server")]
    server: ApiServer,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_model: Option<ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_model: Option<ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tts_model: Option<ModelConfig>,
    extras: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ApiServer {
    #[serde(rename = "type")]
    ty: String,
    version: String,
    #[serde(rename = "ggml_plugin_version")]
    plugin_version: String,
    port: String,
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
    pub ubatch_size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<PromptTemplateType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_predict: Option<i32>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_split: Option<String>,
}
