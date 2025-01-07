#[macro_use]
extern crate log;

mod backend;
mod error;
mod utils;

use anyhow::Result;
use chat_prompts::PromptTemplateType;
use clap::{ArgGroup, Parser};
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
use serde::{de, Deserialize, Serialize};
use std::{collections::HashMap, fs, net::SocketAddr, path::PathBuf};
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
#[command(group = ArgGroup::new("socket_address_group").multiple(false).args(&["socket_addr", "port"]))]
#[command(group = ArgGroup::new("config_group").multiple(false).args(&["config", "prompt_template"]))]
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
    #[arg(short, long, value_delimiter = ',', value_parser = clap::value_parser!(PromptTemplateType), group = "config_group", required = true)]
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
    /// Socket address of LlamaEdge API Server instance. For example, `0.0.0.0:8080`.
    #[arg(long, default_value = None, value_parser = clap::value_parser!(SocketAddr), group = "socket_address_group")]
    socket_addr: Option<SocketAddr>,
    /// Port number
    #[arg(long, default_value = DEFAULT_PORT, value_parser = clap::value_parser!(u16), group = "socket_address_group")]
    port: u16,
    /// Path to the configuration file (*.yaml)
    #[arg(long, group = "config_group")]
    config: Option<PathBuf>,
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

#[derive(Debug)]
struct CliConfig {
    /// Sets names for chat and/or embedding models. To run both chat and embedding models, the names should be separated by comma without space, for example, '--model-name Llama-2-7b,all-minilm'. The first value is for the chat model, and the second is for the embedding model.
    model_name: Vec<String>,
    /// Model aliases for chat and embedding models
    model_alias: Vec<String>,
    /// Sets context sizes for chat and/or embedding models. To run both chat and embedding models, the sizes should be separated by comma without space, for example, '--ctx-size 4096,384'. The first value is for the chat model, and the second is for the embedding model.
    ctx_size: Vec<u64>,
    /// Sets batch sizes for chat and/or embedding models. To run both chat and embedding models, the sizes should be separated by comma without space, for example, '--batch-size 128,64'. The first value is for the chat model, and the second is for the embedding model.
    batch_size: Vec<u64>,
    /// Sets prompt templates for chat and/or embedding models, respectively. To run both chat and embedding models, the prompt templates should be separated by comma without space, for example, '--prompt-template llama-2-chat,embedding'. The first value is for the chat model, and the second is for the embedding model.
    prompt_template: Vec<PromptTemplateType>,
    /// Halt generation at PROMPT, return control.
    reverse_prompt: Option<String>,
    /// Number of tokens to predict
    n_predict: u64,
    /// Number of layers to run on the GPU
    n_gpu_layers: u64,
    /// Split the model across multiple GPUs. Possible values:
    /// - `none`: use one GPU only
    /// - `layer`: split layers and KV across GPUs (default)
    /// - `row`: split rows across GPUs
    split_mode: String,
    /// The main GPU to use.
    main_gpu: Option<u64>,
    /// How split tensors should be distributed accross GPUs. If None the model is not split; otherwise, a comma-separated list of non-negative values, e.g., "3,2" presents 60% of the data to GPU 0 and 40% to GPU 1.
    tensor_split: Option<String>,
    /// Number of threads to use during computation
    threads: u64,
    /// Disable memory mapping for file access of chat models
    no_mmap: bool,
    /// Temperature for sampling
    temp: f64,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 1.0 = disabled
    top_p: f64,
    /// Penalize repeat sequence of tokens
    repeat_penalty: f64,
    /// Repeat alpha presence penalty. 0.0 = disabled
    presence_penalty: f64,
    /// Repeat alpha frequency penalty. 0.0 = disabled
    frequency_penalty: f64,
    /// BNF-like grammar to constrain generations (see samples in grammars/ dir).
    grammar: Option<String>,
    /// JSON schema to constrain generations (<https://json-schema.org/>), e.g. `{}` for any JSON object. For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead.
    json_schema: Option<String>,
    /// Path to the multimodal projector file
    llava_mmproj: Option<String>,
    /// Socket address of LlamaEdge API Server instance. For example, `0.0.0.0:8080`.
    socket_addr: Option<SocketAddr>,
}
impl<'de> Deserialize<'de> for CliConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "kebab-case")]
        enum Field {
            ModelName,
            ModelAlias,
            CtxSize,
            BatchSize,
            PromptTemplate,
            ReversePrompt,
            NPredict,
            NGpuLayers,
            SplitMode,
            MainGpu,
            TensorSplit,
            Threads,
            NoMmap,
            Temp,
            TopP,
            RepeatPenalty,
            PresencePenalty,
            FrequencyPenalty,
            Grammar,
            JsonSchema,
            LlavaMmproj,
            SocketAddr,
        }

        struct CliConfigVisitor;

        impl<'de> serde::de::Visitor<'de> for CliConfigVisitor {
            type Value = CliConfig;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct CliConfig")
            }

            fn visit_map<V>(self, mut map: V) -> Result<CliConfig, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut model_name = None;
                let mut model_alias = None;
                let mut ctx_size = None;
                let mut batch_size = None;
                let mut prompt_template: Option<Vec<String>> = None;
                let mut reverse_prompt = None;
                let mut n_predict = None;
                let mut n_gpu_layers = None;
                let mut split_mode = None;
                let mut main_gpu = None;
                let mut tensor_split = None;
                let mut threads = None;
                let mut no_mmap = None;
                let mut temp = None;
                let mut top_p = None;
                let mut repeat_penalty = None;
                let mut presence_penalty = None;
                let mut frequency_penalty = None;
                let mut grammar: Option<String> = None;
                let mut json_schema = None;
                let mut llava_mmproj = None;
                let mut socket_addr: Option<SocketAddr> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::ModelName => {
                            if model_name.is_some() {
                                return Err(de::Error::duplicate_field("model-name"));
                            }

                            model_name = Some(map.next_value()?)
                        }
                        Field::ModelAlias => {
                            if model_alias.is_some() {
                                return Err(de::Error::duplicate_field("model-alias"));
                            }

                            model_alias = Some(map.next_value()?)
                        }
                        Field::CtxSize => {
                            if ctx_size.is_some() {
                                return Err(de::Error::duplicate_field("ctx-size"));
                            }

                            ctx_size = Some(map.next_value()?)
                        }
                        Field::BatchSize => {
                            if batch_size.is_some() {
                                return Err(de::Error::duplicate_field("batch-size"));
                            }

                            batch_size = Some(map.next_value()?)
                        }
                        Field::PromptTemplate => {
                            if prompt_template.is_some() {
                                return Err(de::Error::duplicate_field("prompt-template"));
                            }

                            prompt_template = Some(map.next_value()?);
                        }
                        Field::ReversePrompt => {
                            if reverse_prompt.is_some() {
                                return Err(de::Error::duplicate_field("reverse-prompt"));
                            }

                            reverse_prompt = Some(map.next_value()?)
                        }
                        Field::NPredict => {
                            if n_predict.is_some() {
                                return Err(de::Error::duplicate_field("n-predict"));
                            }

                            n_predict = Some(map.next_value()?)
                        }
                        Field::NGpuLayers => {
                            if n_gpu_layers.is_some() {
                                return Err(de::Error::duplicate_field("n-gpu-layers"));
                            }

                            n_gpu_layers = Some(map.next_value()?)
                        }
                        Field::SplitMode => {
                            if split_mode.is_some() {
                                return Err(de::Error::duplicate_field("split-mode"));
                            }

                            split_mode = Some(map.next_value()?)
                        }
                        Field::MainGpu => {
                            if main_gpu.is_some() {
                                return Err(de::Error::duplicate_field("main-gpu"));
                            }

                            main_gpu = Some(map.next_value()?)
                        }
                        Field::TensorSplit => {
                            if tensor_split.is_some() {
                                return Err(de::Error::duplicate_field("tensor-split"));
                            }

                            tensor_split = Some(map.next_value()?)
                        }
                        Field::Threads => {
                            if threads.is_some() {
                                return Err(de::Error::duplicate_field("threads"));
                            }

                            threads = Some(map.next_value()?)
                        }
                        Field::NoMmap => {
                            if no_mmap.is_some() {
                                return Err(de::Error::duplicate_field("no-mmap"));
                            }

                            no_mmap = Some(map.next_value()?)
                        }
                        Field::Temp => {
                            if temp.is_some() {
                                return Err(de::Error::duplicate_field("temp"));
                            }

                            temp = Some(map.next_value()?)
                        }
                        Field::TopP => {
                            if top_p.is_some() {
                                return Err(de::Error::duplicate_field("top-p"));
                            }

                            top_p = Some(map.next_value()?)
                        }
                        Field::RepeatPenalty => {
                            if repeat_penalty.is_some() {
                                return Err(de::Error::duplicate_field("repeat-penalty"));
                            }

                            repeat_penalty = Some(map.next_value()?)
                        }
                        Field::PresencePenalty => {
                            if presence_penalty.is_some() {
                                return Err(de::Error::duplicate_field("presence-penalty"));
                            }

                            presence_penalty = Some(map.next_value()?)
                        }
                        Field::FrequencyPenalty => {
                            if frequency_penalty.is_some() {
                                return Err(de::Error::duplicate_field("frequency-penalty"));
                            }

                            frequency_penalty = Some(map.next_value()?)
                        }
                        Field::Grammar => {
                            if grammar.is_some() {
                                return Err(de::Error::duplicate_field("grammar"));
                            }

                            grammar = Some(map.next_value()?)
                        }
                        Field::JsonSchema => {
                            if json_schema.is_some() {
                                return Err(de::Error::duplicate_field("json-schema"));
                            }

                            json_schema = Some(map.next_value()?)
                        }
                        Field::LlavaMmproj => {
                            if llava_mmproj.is_some() {
                                return Err(de::Error::duplicate_field("llava-mmproj"));
                            }

                            llava_mmproj = Some(map.next_value()?)
                        }
                        Field::SocketAddr => {
                            if socket_addr.is_some() {
                                return Err(de::Error::duplicate_field("socket-addr"));
                            }

                            socket_addr = Some(map.next_value()?)
                        }
                    }
                }

                let model_name = model_name
                    .unwrap_or_else(|| vec!["default".to_string(), "embedding".to_string()]);

                let model_alias = model_alias
                    .unwrap_or_else(|| vec!["default".to_string(), "embedding".to_string()]);

                let ctx_size = ctx_size.unwrap_or_else(|| vec![4096, 384]);

                let batch_size = batch_size.unwrap_or_else(|| vec![512, 512]);

                let prompt_template: Vec<PromptTemplateType> = match prompt_template {
                    Some(prompt_template) => {
                        prompt_template.iter().map(|p| p.parse().unwrap()).collect()
                    }
                    None => return Err(de::Error::missing_field("prompt-template")),
                };

                let reverse_prompt = reverse_prompt.unwrap_or_default();

                let n_predict = n_predict.unwrap_or(1024);

                let n_gpu_layers = n_gpu_layers.unwrap_or(100);

                let split_mode = split_mode.unwrap_or("layer".to_string());

                let main_gpu = main_gpu.unwrap();

                let tensor_split = tensor_split.unwrap();

                let threads = threads.unwrap_or(2);

                let no_mmap = no_mmap.unwrap();

                let temp = temp.unwrap_or(1.0);

                let top_p = top_p.unwrap_or(1.0);

                let repeat_penalty = repeat_penalty.unwrap_or(1.1);

                let presence_penalty = presence_penalty.unwrap_or(0.0);

                let frequency_penalty = frequency_penalty.unwrap_or(0.0);

                let json_schema = json_schema.unwrap();

                let llava_mmproj = llava_mmproj.unwrap();

                Ok(CliConfig {
                    model_name,
                    model_alias,
                    ctx_size,
                    batch_size,
                    prompt_template,
                    reverse_prompt,
                    n_predict,
                    n_gpu_layers,
                    split_mode,
                    main_gpu,
                    tensor_split,
                    threads,
                    no_mmap,
                    temp,
                    top_p,
                    repeat_penalty,
                    presence_penalty,
                    frequency_penalty,
                    grammar,
                    json_schema,
                    llava_mmproj,
                    socket_addr,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "model-name",
            "model-alias",
            "ctx-size",
            "batch-size",
            "prompt-template",
            "reverse-prompt",
            "n-predict",
            "n-gpu-layers",
            "split-mode",
            "main-gpu",
            "tensor-split",
            "threads",
            "no-mmap",
            "temp",
            "top-p",
            "repeat-penalty",
            "presence-penalty",
            "frequency-penalty",
            "grammar",
            "json-schema",
            "llava-mmproj",
            "socket-addr",
        ];

        deserializer.deserialize_struct("CliConfig", FIELDS, CliConfigVisitor)
    }
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

    info!(target: "stdout", "log_level: {}", log_level);

    if let Ok(api_key) = std::env::var("API_KEY") {
        // define a const variable for the API key
        if let Err(e) = LLAMA_API_KEY.set(api_key) {
            let err_msg = format!("Failed to set API key. {}", e);

            error!(target: "stdout", "{}", err_msg);

            return Err(ServerError::Operation(err_msg));
        }
    }

    // parse the command line arguments
    let mut cli = Cli::parse();

    // log the version of the server
    info!(target: "stdout", "server version: {}", env!("CARGO_PKG_VERSION"));

    if let Some(config_file) = &cli.config {
        match config_file.exists() {
            true => {
                let yaml_content = match fs::read_to_string(config_file) {
                    Ok(yaml_content) => yaml_content,
                    Err(e) => {
                        let err_msg = format!("Failed to read config file: {}", e);

                        error!(target: "stdout", "{}", err_msg);

                        return Err(ServerError::Operation(err_msg));
                    }
                };

                let config: CliConfig = match serde_yaml::from_str(&yaml_content) {
                    Ok(config) => config,
                    Err(e) => {
                        let err_msg = format!("Failed to parse YAML: {}", e);

                        error!(target: "stdout", "{}", err_msg);

                        return Err(ServerError::Operation(err_msg));
                    }
                };

                debug!(target: "stdout", "config: {:?}", &config);

                // update the CLI arguments with the config file
                cli.model_name = config.model_name;
                cli.model_alias = config.model_alias;
                cli.ctx_size = config.ctx_size;
                cli.batch_size = config.batch_size;
                cli.prompt_template = config.prompt_template;
                cli.reverse_prompt = config.reverse_prompt;
                cli.n_predict = config.n_predict;
                cli.n_gpu_layers = config.n_gpu_layers;
                cli.split_mode = config.split_mode;
                cli.main_gpu = config.main_gpu;
                cli.tensor_split = config.tensor_split;
                cli.threads = config.threads;
                cli.no_mmap = Some(config.no_mmap);
                cli.temp = config.temp;
                cli.top_p = config.top_p;
                cli.repeat_penalty = config.repeat_penalty;
                cli.presence_penalty = config.presence_penalty;
                cli.frequency_penalty = config.frequency_penalty;
                cli.grammar = config.grammar.unwrap_or_default();
                cli.json_schema = config.json_schema;
                cli.llava_mmproj = config.llava_mmproj;
                cli.socket_addr = config.socket_addr;
            }
            false => {
                let err_msg = format!("Config file not found: {}", config_file.display());

                error!(target: "stdout", "{}", err_msg);

                return Err(ServerError::Operation(err_msg));
            }
        }
    }

    // log model names
    if cli.model_name.is_empty() && cli.model_name.len() > 2 {
        return Err(ServerError::ArgumentError(
            "Invalid setting for model name. For running chat or embedding model, please specify a single model name. For running both chat and embedding models, please specify two model names: the first one for chat model, the other for embedding model.".to_owned(),
        ));
    }
    info!(target: "stdout", "model_name: {}", cli.model_name.join(",").to_string());

    // log model alias
    let mut model_alias = String::new();
    if cli.model_name.len() == 1 {
        model_alias.clone_from(&cli.model_alias[0]);
    } else if cli.model_alias.len() == 2 {
        model_alias = cli.model_alias.join(",").to_string();
    }
    info!(target: "stdout", "model_alias: {}", model_alias);

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
    info!(target: "stdout", "ctx_size: {}", ctx_sizes_str);

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
    info!(target: "stdout", "batch_size: {}", batch_sizes_str);

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
    info!(target: "stdout", "prompt_template: {}", prompt_template_str);
    if cli.model_name.len() != cli.prompt_template.len() {
        return Err(ServerError::ArgumentError(
            "The number of model names and prompt templates must be the same.".to_owned(),
        ));
    }

    // log reverse prompt
    if let Some(reverse_prompt) = &cli.reverse_prompt {
        info!(target: "stdout", "reverse_prompt: {}", reverse_prompt);
    }

    // log n_predict
    info!(target: "stdout", "n_predict: {}", cli.n_predict);

    // log n_gpu_layers
    info!(target: "stdout", "n_gpu_layers: {}", cli.n_gpu_layers);

    // log split_mode
    info!(target: "stdout", "split_mode: {}", cli.split_mode);

    // log main_gpu
    if let Some(main_gpu) = &cli.main_gpu {
        info!(target: "stdout", "main_gpu: {}", main_gpu);
    }

    // log tensor_split
    if let Some(tensor_split) = &cli.tensor_split {
        info!(target: "stdout", "tensor_split: {}", tensor_split);
    }

    // log threads
    info!(target: "stdout", "threads: {}", cli.threads);

    // log no_mmap
    if let Some(no_mmap) = &cli.no_mmap {
        info!(target: "stdout", "no_mmap: {}", no_mmap);
    }

    // log temperature
    info!(target: "stdout", "temp: {}", cli.temp);

    // log top-p sampling
    info!(target: "stdout", "top_p: {}", cli.top_p);

    // repeat penalty
    info!(target: "stdout", "repeat_penalty: {}", cli.repeat_penalty);

    // log presence penalty
    info!(target: "stdout", "presence_penalty: {}", cli.presence_penalty);

    // log frequency penalty
    info!(target: "stdout", "frequency_penalty: {}", cli.frequency_penalty);

    // log grammar
    if !cli.grammar.is_empty() {
        info!(target: "stdout", "grammar: {}", &cli.grammar);
    }

    // log json schema
    if let Some(json_schema) = &cli.json_schema {
        info!(target: "stdout", "json_schema: {}", json_schema);
    }

    // log multimodal projector
    if let Some(llava_mmproj) = &cli.llava_mmproj {
        info!(target: "stdout", "llava_mmproj: {}", llava_mmproj);
    }

    // initialize the core context
    let mut chat_model_config = None;
    let mut embedding_model_config = None;
    if cli.prompt_template.len() == 1 {
        match cli.prompt_template[0] {
            PromptTemplateType::Embedding => {
                // create a Metadata instance
                let metadata_embedding = GgmlMetadataBuilder::new(
                    cli.model_name[0].clone(),
                    cli.model_alias[0].clone(),
                    cli.prompt_template[0],
                )
                .with_ctx_size(cli.ctx_size[0])
                .with_batch_size(cli.batch_size[0])
                .with_split_mode(cli.split_mode)
                .with_main_gpu(cli.main_gpu)
                .with_tensor_split(cli.tensor_split)
                .with_threads(cli.threads)
                .enable_plugin_log(true)
                .enable_debug_log(plugin_debug)
                .build();

                // set the embedding model config
                embedding_model_config = Some(ModelConfig {
                    name: metadata_embedding.model_name.clone(),
                    ty: "embedding".to_string(),
                    ctx_size: metadata_embedding.ctx_size,
                    batch_size: metadata_embedding.batch_size,
                    prompt_template: Some(PromptTemplateType::Embedding),
                    n_predict: Some(cli.n_predict),
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
                llama_core::init_ggml_context(None, Some(&[metadata_embedding]))
                    .map_err(|e| ServerError::Operation(format!("{}", e)))?;
            }
            _ => {
                // create a Metadata instance
                let metadata_chat = GgmlMetadataBuilder::new(
                    cli.model_name[0].clone(),
                    cli.model_alias[0].clone(),
                    cli.prompt_template[0],
                )
                .with_ctx_size(cli.ctx_size[0])
                .with_batch_size(cli.batch_size[0])
                .with_n_predict(cli.n_predict)
                .with_n_gpu_layers(cli.n_gpu_layers)
                .with_split_mode(cli.split_mode)
                .with_main_gpu(cli.main_gpu)
                .with_tensor_split(cli.tensor_split)
                .with_threads(cli.threads)
                .disable_mmap(cli.no_mmap)
                .with_temperature(cli.temp)
                .with_top_p(cli.top_p)
                .with_repeat_penalty(cli.repeat_penalty)
                .with_presence_penalty(cli.presence_penalty)
                .with_frequency_penalty(cli.frequency_penalty)
                .with_grammar(cli.grammar)
                .with_json_schema(cli.json_schema)
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
                    split_mode: Some(metadata_chat.split_mode.clone()),
                    main_gpu: metadata_chat.main_gpu,
                    tensor_split: metadata_chat.tensor_split.clone(),
                });

                // initialize the core context
                llama_core::init_ggml_context(Some(&[metadata_chat]), None)
                    .map_err(|e| ServerError::Operation(format!("{}", e)))?;
            }
        }
    } else if cli.prompt_template.len() == 2 {
        // create a Metadata instance
        let metadata_chat = GgmlMetadataBuilder::new(
            cli.model_name[0].clone(),
            cli.model_alias[0].clone(),
            cli.prompt_template[0],
        )
        .with_ctx_size(cli.ctx_size[0])
        .with_batch_size(cli.batch_size[0])
        .with_n_predict(cli.n_predict)
        .with_n_gpu_layers(cli.n_gpu_layers)
        .with_split_mode(cli.split_mode.clone())
        .with_main_gpu(cli.main_gpu)
        .with_tensor_split(cli.tensor_split.clone())
        .with_threads(cli.threads)
        .disable_mmap(cli.no_mmap)
        .with_temperature(cli.temp)
        .with_top_p(cli.top_p)
        .with_repeat_penalty(cli.repeat_penalty)
        .with_presence_penalty(cli.presence_penalty)
        .with_frequency_penalty(cli.frequency_penalty)
        .with_grammar(cli.grammar)
        .with_json_schema(cli.json_schema)
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
            split_mode: Some(metadata_chat.split_mode.clone()),
            main_gpu: metadata_chat.main_gpu,
            tensor_split: metadata_chat.tensor_split.clone(),
        });

        // create a Metadata instance
        let metadata_embedding = GgmlMetadataBuilder::new(
            cli.model_name[1].clone(),
            cli.model_alias[1].clone(),
            cli.prompt_template[1],
        )
        .with_ctx_size(cli.ctx_size[1])
        .with_batch_size(cli.batch_size[1])
        .with_split_mode(cli.split_mode)
        .with_main_gpu(cli.main_gpu)
        .with_tensor_split(cli.tensor_split)
        .with_threads(cli.threads)
        .enable_plugin_log(true)
        .enable_debug_log(plugin_debug)
        .build();

        // set the embedding model config
        embedding_model_config = Some(ModelConfig {
            name: metadata_embedding.model_name.clone(),
            ty: "embedding".to_string(),
            ctx_size: metadata_embedding.ctx_size,
            batch_size: metadata_embedding.batch_size,
            prompt_template: Some(PromptTemplateType::Embedding),
            n_predict: Some(cli.n_predict),
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
        llama_core::init_ggml_context(Some(&[metadata_chat]), Some(&[metadata_embedding]))
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
    let addr = match cli.socket_addr {
        Some(addr) => addr,
        None => SocketAddr::from(([0, 0, 0, 0], cli.port)),
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
        extras: HashMap::new(),
    };
    SERVER_INFO
        .set(server_info)
        .map_err(|_| ServerError::Operation("Failed to set `SERVER_INFO`.".to_string()))?;

    let new_service = make_service_fn(move |conn: &AddrStream| {
        // log socket address
        info!(target: "stdout", "remote_addr: {}, local_addr: {}", conn.remote_addr().to_string(), conn.local_addr().to_string());

        // web ui
        let web_ui = cli.web_ui.to_string_lossy().to_string();

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_split: Option<String>,
}
