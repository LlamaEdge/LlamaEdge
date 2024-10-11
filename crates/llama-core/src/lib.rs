//! Llama Core, abbreviated as `llama-core`, defines a set of APIs. Developers can utilize these APIs to build applications based on large models, such as chatbots, RAG, and more.

#[cfg(feature = "logging")]
#[macro_use]
extern crate log;

pub mod audio;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod graph;
pub mod images;
pub mod models;
pub mod rag;
#[cfg(feature = "search")]
pub mod search;
pub mod utils;

pub use error::LlamaCoreError;
pub use graph::{EngineType, Graph, GraphBuilder};

use chat_prompts::PromptTemplateType;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::Path,
    sync::{Mutex, RwLock},
};
use utils::get_output_buffer;
use wasmedge_stable_diffusion::*;

// key: model_name, value: Graph
pub(crate) static CHAT_GRAPHS: OnceCell<Mutex<HashMap<String, Graph<GgmlMetadata>>>> =
    OnceCell::new();
// key: model_name, value: Graph
pub(crate) static EMBEDDING_GRAPHS: OnceCell<Mutex<HashMap<String, Graph<GgmlMetadata>>>> =
    OnceCell::new();
// cache bytes for decoding utf8
pub(crate) static CACHED_UTF8_ENCODINGS: OnceCell<Mutex<Vec<u8>>> = OnceCell::new();
// running mode
pub(crate) static RUNNING_MODE: OnceCell<RwLock<RunningMode>> = OnceCell::new();
// stable diffusion context for the text-to-image task
pub(crate) static SD_TEXT_TO_IMAGE: OnceCell<Mutex<TextToImage>> = OnceCell::new();
// stable diffusion context for the image-to-image task
pub(crate) static SD_IMAGE_TO_IMAGE: OnceCell<Mutex<ImageToImage>> = OnceCell::new();
// context for the audio task
pub(crate) static AUDIO_GRAPH: OnceCell<Mutex<Graph<WhisperMetadata>>> = OnceCell::new();
// context for the piper task
pub(crate) static PIPER_GRAPH: OnceCell<Mutex<Graph<GgmlMetadata>>> = OnceCell::new();

pub(crate) const MAX_BUFFER_SIZE: usize = 2usize.pow(14) * 15 + 128;
pub(crate) const OUTPUT_TENSOR: usize = 0;
const PLUGIN_VERSION: usize = 1;

pub trait BaseMetadata {
    fn model_name(&self) -> &str;
    fn model_alias(&self) -> &str;
    fn prompt_template(&self) -> PromptTemplateType;
}

/// Model metadata
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GgmlMetadata {
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub model_name: String,
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub model_alias: String,
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub log_prompts: bool,
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub prompt_template: PromptTemplateType,

    // * Plugin parameters (used by this plugin):
    #[serde(rename = "enable-log")]
    pub log_enable: bool,
    #[serde(rename = "enable-debug-log")]
    pub debug_log: bool,
    // #[serde(rename = "stream-stdout")]
    // pub stream_stdout: bool,
    #[serde(rename = "embedding")]
    pub embeddings: bool,
    #[serde(rename = "n-predict")]
    pub n_predict: u64,
    #[serde(skip_serializing_if = "Option::is_none", rename = "reverse-prompt")]
    pub reverse_prompt: Option<String>,
    /// path to the multimodal projector file for llava
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    /// Path to the image file for llava
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,

    // * Model parameters (need to reload the model if updated):
    #[serde(rename = "n-gpu-layers")]
    pub n_gpu_layers: u64,
    /// The main GPU to use. Defaults to None.
    #[serde(rename = "main-gpu")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<u64>,
    /// How split tensors should be distributed accross GPUs. If None the model is not split; otherwise, a comma-separated list of non-negative values, e.g., "3,2" presents 60% of the data to GPU 0 and 40% to GPU 1.
    #[serde(rename = "tensor-split")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_split: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "use-mmap")]
    pub use_mmap: Option<bool>,
    // * Context parameters (used by the llama context):
    #[serde(rename = "ctx-size")]
    pub ctx_size: u64,
    #[serde(rename = "batch-size")]
    pub batch_size: u64,
    #[serde(rename = "threads")]
    pub threads: u64,

    // * Sampling parameters (used by the llama sampling context).
    #[serde(rename = "temp")]
    pub temperature: f64,
    #[serde(rename = "top-p")]
    pub top_p: f64,
    #[serde(rename = "repeat-penalty")]
    pub repeat_penalty: f64,
    #[serde(rename = "presence-penalty")]
    pub presence_penalty: f64,
    #[serde(rename = "frequency-penalty")]
    pub frequency_penalty: f64,

    // * grammar parameters
    /// BNF-like grammar to constrain generations (see samples in grammars/ dir). Defaults to empty string.
    pub grammar: String,
    /// JSON schema to constrain generations (<https://json-schema.org/>), e.g. `{}` for any JSON object. For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<String>,

    // * parameters for whisper
    pub translate: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Number of processors to use during computation. Defaults to 1.
    pub processors: u32,
    /// Time offset in milliseconds. Defaults to 0.
    pub offset_t: u32,
    /// Duration of audio to process in milliseconds. Defaults to 0.
    pub duration: u32,
    /// Maximum number of text context tokens to store. Defaults to -1.
    pub max_context: i32,
    /// Maximum segment length in characters. Defaults to 0.
    pub max_len: u32,
    /// Split on word rather than on token. Defaults to false.
    pub split_on_word: bool,
    /// Output result in a text file. Defaults to false.
    pub output_txt: bool,
    /// Output result in a vtt file. Defaults to false.
    pub output_vtt: bool,
    /// Output result in a srt file. Defaults to false.
    pub output_srt: bool,
    /// Output result in a lrc file. Defaults to false.
    pub output_lrc: bool,
    /// Output result in a CSV file. Defaults to false.
    pub output_csv: bool,
    /// Output result in a JSON file. Defaults to false.
    pub output_json: bool,
}
impl Default for GgmlMetadata {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_alias: String::new(),
            log_prompts: false,
            debug_log: false,
            prompt_template: PromptTemplateType::Llama2Chat,
            log_enable: false,
            embeddings: false,
            n_predict: 1024,
            reverse_prompt: None,
            mmproj: None,
            image: None,
            n_gpu_layers: 100,
            main_gpu: None,
            tensor_split: None,
            use_mmap: Some(true),
            ctx_size: 512,
            batch_size: 512,
            threads: 2,
            temperature: 1.0,
            top_p: 1.0,
            repeat_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            grammar: String::new(),
            json_schema: None,
            translate: false,
            language: None,
            processors: 1,
            offset_t: 0,
            duration: 0,
            max_context: -1,
            max_len: 0,
            split_on_word: false,
            output_txt: false,
            output_vtt: false,
            output_srt: false,
            output_lrc: false,
            output_csv: false,
            output_json: false,
        }
    }
}
impl BaseMetadata for GgmlMetadata {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn model_alias(&self) -> &str {
        &self.model_alias
    }

    fn prompt_template(&self) -> PromptTemplateType {
        self.prompt_template
    }
}

/// Builder for the `Metadata` struct
#[derive(Debug)]
pub struct GgmlMetadataBuilder {
    metadata: GgmlMetadata,
}
impl GgmlMetadataBuilder {
    pub fn new<S: Into<String>>(model_name: S, model_alias: S, pt: PromptTemplateType) -> Self {
        let metadata = GgmlMetadata {
            model_name: model_name.into(),
            model_alias: model_alias.into(),
            prompt_template: pt,
            ..Default::default()
        };

        Self { metadata }
    }

    pub fn with_prompt_template(mut self, template: PromptTemplateType) -> Self {
        self.metadata.prompt_template = template;
        self
    }

    pub fn enable_plugin_log(mut self, enable: bool) -> Self {
        self.metadata.log_enable = enable;
        self
    }

    pub fn enable_debug_log(mut self, enable: bool) -> Self {
        self.metadata.debug_log = enable;
        self
    }

    pub fn enable_prompts_log(mut self, enable: bool) -> Self {
        self.metadata.log_prompts = enable;
        self
    }

    pub fn enable_embeddings(mut self, enable: bool) -> Self {
        self.metadata.embeddings = enable;
        self
    }

    pub fn with_n_predict(mut self, n: u64) -> Self {
        self.metadata.n_predict = n;
        self
    }

    pub fn with_main_gpu(mut self, gpu: Option<u64>) -> Self {
        self.metadata.main_gpu = gpu;
        self
    }

    pub fn with_tensor_split(mut self, split: Option<String>) -> Self {
        self.metadata.tensor_split = split;
        self
    }

    pub fn with_threads(mut self, threads: u64) -> Self {
        self.metadata.threads = threads;
        self
    }

    pub fn with_reverse_prompt(mut self, prompt: Option<String>) -> Self {
        self.metadata.reverse_prompt = prompt;
        self
    }

    pub fn with_mmproj(mut self, path: Option<String>) -> Self {
        self.metadata.mmproj = path;
        self
    }

    pub fn with_image(mut self, path: impl Into<String>) -> Self {
        self.metadata.image = Some(path.into());
        self
    }

    pub fn with_n_gpu_layers(mut self, n: u64) -> Self {
        self.metadata.n_gpu_layers = n;
        self
    }

    pub fn disable_mmap(mut self, disable: Option<bool>) -> Self {
        self.metadata.use_mmap = disable.map(|v| !v);
        self
    }

    pub fn with_ctx_size(mut self, size: u64) -> Self {
        self.metadata.ctx_size = size;
        self
    }

    pub fn with_batch_size(mut self, size: u64) -> Self {
        self.metadata.batch_size = size;
        self
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.metadata.temperature = temp;
        self
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.metadata.top_p = top_p;
        self
    }

    pub fn with_repeat_penalty(mut self, penalty: f64) -> Self {
        self.metadata.repeat_penalty = penalty;
        self
    }

    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        self.metadata.presence_penalty = penalty;
        self
    }

    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
        self.metadata.frequency_penalty = penalty;
        self
    }

    pub fn with_grammar(mut self, grammar: impl Into<String>) -> Self {
        self.metadata.grammar = grammar.into();
        self
    }

    pub fn with_json_schema(mut self, schema: Option<String>) -> Self {
        self.metadata.json_schema = schema;
        self
    }

    pub fn build(self) -> GgmlMetadata {
        self.metadata
    }
}

/// Metadata for whisper model
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WhisperMetadata {
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub model_name: String,
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub model_alias: String,

    /// Enable debug mode. Defaults to false.
    pub debug_mode: bool,
    /// Number of threads to use during computation. Defaults to 4.
    pub threads: u64,
    /// Translate from source language to english. Defaults to false.
    pub translate: bool,
    /// The language of the input audio. `auto` for auto-detection. Defaults to `en`.
    ///
    /// Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will improve accuracy and latency.
    pub language: String,
    /// Number of processors to use during computation. Defaults to 1.
    pub processors: u32,
    /// Time offset in milliseconds. Defaults to 0.
    pub offset_t: u32,
    /// Duration of audio to process in milliseconds. Defaults to 0.
    pub duration: u32,
    /// Maximum number of text context tokens to store. Defaults to -1.
    pub max_context: i32,
    /// Maximum segment length in characters. Defaults to 0.
    pub max_len: u32,
    /// Split on word rather than on token. Defaults to false.
    pub split_on_word: bool,
    /// Output result in a text file. Defaults to false.
    pub output_txt: bool,
    /// Output result in a vtt file. Defaults to false.
    pub output_vtt: bool,
    /// Output result in a srt file. Defaults to false.
    pub output_srt: bool,
    /// Output result in a lrc file. Defaults to false.
    pub output_lrc: bool,
    /// Output result in a CSV file. Defaults to false.
    pub output_csv: bool,
    /// Output result in a JSON file. Defaults to false.
    pub output_json: bool,
    /// The sampling temperature, between 0 and 1. Defaults to 0.00.
    pub temperature: f64,
}
impl Default for WhisperMetadata {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_alias: String::new(),
            debug_mode: false,
            threads: 4,
            translate: false,
            language: "en".to_string(),
            processors: 1,
            offset_t: 0,
            duration: 0,
            max_context: -1,
            max_len: 0,
            split_on_word: false,
            output_txt: false,
            output_vtt: false,
            output_srt: false,
            output_lrc: false,
            output_csv: false,
            output_json: false,
            temperature: 0.0,
        }
    }
}
impl BaseMetadata for WhisperMetadata {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn model_alias(&self) -> &str {
        &self.model_alias
    }

    fn prompt_template(&self) -> PromptTemplateType {
        PromptTemplateType::Null
    }
}

/// Builder for creating an audio metadata
#[derive(Debug)]
pub struct WhisperMetadataBuilder {
    metadata: WhisperMetadata,
}
impl WhisperMetadataBuilder {
    pub fn new<S: Into<String>>(model_name: S, model_alias: S) -> Self {
        let metadata = WhisperMetadata {
            model_name: model_name.into(),
            model_alias: model_alias.into(),
            ..Default::default()
        };

        Self { metadata }
    }

    pub fn enable_debug(mut self, enable: bool) -> Self {
        self.metadata.debug_mode = enable;
        self
    }

    pub fn with_threads(mut self, threads: u64) -> Self {
        self.metadata.threads = threads;
        self
    }

    pub fn enable_translate(mut self, enable: bool) -> Self {
        self.metadata.translate = enable;
        self
    }

    pub fn with_language(mut self, language: String) -> Self {
        self.metadata.language = language;
        self
    }

    pub fn with_processors(mut self, processors: u32) -> Self {
        self.metadata.processors = processors;
        self
    }

    pub fn with_offset_t(mut self, offset_t: u32) -> Self {
        self.metadata.offset_t = offset_t;
        self
    }

    pub fn with_duration(mut self, duration: u32) -> Self {
        self.metadata.duration = duration;
        self
    }

    pub fn with_max_context(mut self, max_context: i32) -> Self {
        self.metadata.max_context = max_context;
        self
    }

    pub fn with_max_len(mut self, max_len: u32) -> Self {
        self.metadata.max_len = max_len;
        self
    }

    pub fn split_on_word(mut self, split_on_word: bool) -> Self {
        self.metadata.split_on_word = split_on_word;
        self
    }

    pub fn output_txt(mut self, output_txt: bool) -> Self {
        self.metadata.output_txt = output_txt;
        self
    }

    pub fn output_vtt(mut self, output_vtt: bool) -> Self {
        self.metadata.output_vtt = output_vtt;
        self
    }

    pub fn output_srt(mut self, output_srt: bool) -> Self {
        self.metadata.output_srt = output_srt;
        self
    }

    pub fn output_lrc(mut self, output_lrc: bool) -> Self {
        self.metadata.output_lrc = output_lrc;
        self
    }

    pub fn output_csv(mut self, output_csv: bool) -> Self {
        self.metadata.output_csv = output_csv;
        self
    }

    pub fn output_json(mut self, output_json: bool) -> Self {
        self.metadata.output_json = output_json;
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.metadata.temperature = temperature;
        self
    }

    pub fn build(self) -> WhisperMetadata {
        self.metadata
    }
}

/// Initialize the core context
pub fn init_core_context(
    metadata_for_chats: Option<&[GgmlMetadata]>,
    metadata_for_embeddings: Option<&[GgmlMetadata]>,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the core context");

    if metadata_for_chats.is_none() && metadata_for_embeddings.is_none() {
        let err_msg = "Failed to initialize the core context. Please set metadata for chat completions and/or embeddings.";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::InitContext(err_msg.into()));
    }

    let mut mode = RunningMode::Embeddings;

    if let Some(metadata_chats) = metadata_for_chats {
        let mut chat_graphs = HashMap::new();
        for metadata in metadata_chats {
            let graph = Graph::new(metadata.clone())?;

            chat_graphs.insert(graph.name().to_string(), graph);
        }
        CHAT_GRAPHS.set(Mutex::new(chat_graphs)).map_err(|_| {
            let err_msg = "Failed to initialize the core context. Reason: The `CHAT_GRAPHS` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

        mode = RunningMode::Chat
    }

    if let Some(metadata_embeddings) = metadata_for_embeddings {
        let mut embedding_graphs = HashMap::new();
        for metadata in metadata_embeddings {
            let graph = Graph::new(metadata.clone())?;

            embedding_graphs.insert(graph.name().to_string(), graph);
        }
        EMBEDDING_GRAPHS
            .set(Mutex::new(embedding_graphs))
            .map_err(|_| {
                let err_msg = "Failed to initialize the core context. Reason: The `EMBEDDING_GRAPHS` has already been initialized";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg.into())
            })?;

        if mode == RunningMode::Chat {
            mode = RunningMode::ChatEmbedding;
        }
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "running mode: {}", mode);

    RUNNING_MODE.set(RwLock::new(mode)).map_err(|_| {
        let err_msg = "Failed to initialize the core context. Reason: The `RUNNING_MODE` has already been initialized";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        LlamaCoreError::InitContext(err_msg.into())
    })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "The core context has been initialized");

    Ok(())
}

/// Initialize the core context for RAG scenarios.
pub fn init_rag_core_context(
    metadata_for_chats: &[GgmlMetadata],
    metadata_for_embeddings: &[GgmlMetadata],
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the core context for RAG scenarios");

    // chat models
    if metadata_for_chats.is_empty() {
        let err_msg = "The metadata for chat models is empty";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::InitContext(err_msg.into()));
    }
    let mut chat_graphs = HashMap::new();
    for metadata in metadata_for_chats {
        let graph = Graph::new(metadata.clone())?;

        chat_graphs.insert(graph.name().to_string(), graph);
    }
    CHAT_GRAPHS.set(Mutex::new(chat_graphs)).map_err(|_| {
        let err_msg = "Failed to initialize the core context. Reason: The `CHAT_GRAPHS` has already been initialized";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        LlamaCoreError::InitContext(err_msg.into())
    })?;

    // embedding models
    if metadata_for_embeddings.is_empty() {
        let err_msg = "The metadata for embeddings is empty";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::InitContext(err_msg.into()));
    }
    let mut embedding_graphs = HashMap::new();
    for metadata in metadata_for_embeddings {
        let graph = Graph::new(metadata.clone())?;

        embedding_graphs.insert(graph.name().to_string(), graph);
    }
    EMBEDDING_GRAPHS
        .set(Mutex::new(embedding_graphs))
        .map_err(|_| {
            let err_msg = "Failed to initialize the core context. Reason: The `EMBEDDING_GRAPHS` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

    let running_mode = RunningMode::Rag;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "running mode: {}", running_mode);

    // set running mode
    RUNNING_MODE.set(RwLock::new(running_mode)).map_err(|_| {
            let err_msg = "Failed to initialize the core context. Reason: The `RUNNING_MODE` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "The core context for RAG scenarios has been initialized");

    Ok(())
}

/// Get the plugin info
///
/// Note that it is required to call `init_core_context` before calling this function.
pub fn get_plugin_info() -> Result<PluginInfo, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Getting the plugin info");

    match running_mode()? {
        RunningMode::Embeddings => {
            let embedding_graphs = match EMBEDDING_GRAPHS.get() {
                Some(embedding_graphs) => embedding_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `EMBEDDING_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let embedding_graphs = embedding_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            let graph = match embedding_graphs.values().next() {
                Some(graph) => graph,
                None => {
                    let err_msg = "Fail to get the underlying value of `EMBEDDING_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            get_plugin_info_by_graph(graph)
        }
        _ => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            let graph = match chat_graphs.values().next() {
                Some(graph) => graph,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            get_plugin_info_by_graph(graph)
        }
    }
}

fn get_plugin_info_by_graph<M: BaseMetadata + serde::Serialize + Clone + Default>(
    graph: &Graph<M>,
) -> Result<PluginInfo, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Getting the plugin info by the graph named {}", graph.name());

    // get the plugin metadata
    let output_buffer = get_output_buffer(graph, PLUGIN_VERSION)?;
    let metadata: serde_json::Value = serde_json::from_slice(&output_buffer[..]).map_err(|e| {
        let err_msg = format!("Fail to deserialize the plugin metadata. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    // get build number of the plugin
    let plugin_build_number = match metadata.get("llama_build_number") {
        Some(value) => match value.as_u64() {
            Some(number) => number,
            None => {
                let err_msg = "Failed to convert the build number of the plugin to u64";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::Operation(err_msg.into()));
            }
        },
        None => {
            let err_msg = "Metadata does not have the field `llama_build_number`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    // get commit id of the plugin
    let plugin_commit = match metadata.get("llama_commit") {
        Some(value) => match value.as_str() {
            Some(commit) => commit,
            None => {
                let err_msg = "Failed to convert the commit id of the plugin to string";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::Operation(err_msg.into()));
            }
        },
        None => {
            let err_msg = "Metadata does not have the field `llama_commit`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Plugin info: b{}(commit {})", plugin_build_number, plugin_commit);

    Ok(PluginInfo {
        build_number: plugin_build_number,
        commit_id: plugin_commit.to_string(),
    })
}

/// Version info of the `wasi-nn_ggml` plugin, including the build number and the commit id.
#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub build_number: u64,
    pub commit_id: String,
}
impl std::fmt::Display for PluginInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "wasinn-ggml plugin: b{}(commit {})",
            self.build_number, self.commit_id
        )
    }
}

/// Running mode
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RunningMode {
    Chat,
    Embeddings,
    ChatEmbedding,
    Rag,
}
impl std::fmt::Display for RunningMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunningMode::Chat => write!(f, "chat"),
            RunningMode::Embeddings => write!(f, "embeddings"),
            RunningMode::ChatEmbedding => write!(f, "chat-embeddings"),
            RunningMode::Rag => write!(f, "rag"),
        }
    }
}

/// Return the current running mode.
pub fn running_mode() -> Result<RunningMode, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get the running mode.");

    let mode = match RUNNING_MODE.get() {
        Some(mode) => match mode.read() {
            Ok(mode) => mode.to_owned(),
            Err(e) => {
                let err_msg = format!("Fail to get the underlying value of `RUNNING_MODE`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::Operation(err_msg));
            }
        },
        None => {
            let err_msg = "Fail to get the underlying value of `RUNNING_MODE`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "running mode: {}", &mode);

    Ok(mode.to_owned())
}

/// Initialize the stable diffusion context with the given full diffusion model
///
/// # Arguments
///
/// * `model_file` - Path to the stable diffusion model file.
///
/// * `ctx` - The context type to create.
pub fn init_sd_context_with_full_model(
    model_file: impl AsRef<str>,
    ctx: SDContextType,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the stable diffusion context with the full model");

    // create the stable diffusion context for the text-to-image task
    if ctx == SDContextType::Full || ctx == SDContextType::TextToImage {
        let sd = StableDiffusion::new(Task::TextToImage, model_file.as_ref());

        let ctx = sd.create_context().map_err(|e| {
            let err_msg = format!("Fail to create the context. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::InitContext(err_msg)
        })?;

        let ctx = match ctx {
            Context::TextToImage(ctx) => ctx,
            _ => {
                let err_msg = "Fail to get the context for the text-to-image task";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::InitContext(err_msg.into()));
            }
        };

        SD_TEXT_TO_IMAGE.set(Mutex::new(ctx)).map_err(|_| {
        let err_msg = "Failed to initialize the stable diffusion context. Reason: The `SD_TEXT_TO_IMAGE` has already been initialized";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        LlamaCoreError::InitContext(err_msg.into())
    })?;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "The stable diffusion text-to-image context has been initialized");
    }

    // create the stable diffusion context for the image-to-image task
    if ctx == SDContextType::Full || ctx == SDContextType::ImageToImage {
        let sd = StableDiffusion::new(Task::ImageToImage, model_file.as_ref());

        let ctx = sd.create_context().map_err(|e| {
            let err_msg = format!("Fail to create the context. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::InitContext(err_msg)
        })?;

        let ctx = match ctx {
            Context::ImageToImage(ctx) => ctx,
            _ => {
                let err_msg = "Fail to get the context for the image-to-image task";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::InitContext(err_msg.into()));
            }
        };

        SD_IMAGE_TO_IMAGE.set(Mutex::new(ctx)).map_err(|_| {
            let err_msg = "Failed to initialize the stable diffusion context. Reason: The `SD_IMAGE_TO_IMAGE` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "The stable diffusion image-to-image context has been initialized");
    }

    Ok(())
}

/// Initialize the stable diffusion context with the given standalone diffusion model
///
/// # Arguments
///
/// * `model_file` - Path to the standalone diffusion model file.
///
/// * `vae` - Path to the VAE model file.
///
/// * `clip_l` - Path to the CLIP model file.
///
/// * `t5xxl` - Path to the T5-XXL model file.
///
/// * `lora_model_dir` - Path to the Lora model directory.
///
/// * `n_threads` - Number of threads to use.
///
/// * `ctx` - The context type to create.
pub fn init_sd_context_with_standalone_model(
    model_file: impl AsRef<str>,
    vae: impl AsRef<str>,
    clip_l: impl AsRef<str>,
    t5xxl: impl AsRef<str>,
    lora_model_dir: impl AsRef<str>,
    n_threads: i32,
    ctx: SDContextType,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the stable diffusion context with the standalone diffusion model");

    // create the stable diffusion context for the text-to-image task
    if ctx == SDContextType::Full || ctx == SDContextType::TextToImage {
        let sd = SDBuidler::new_with_standalone_model(Task::TextToImage, model_file.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_vae_path(vae.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_clip_l_path(clip_l.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_t5xxl_path(t5xxl.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_lora_model_dir(lora_model_dir.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_n_threads(n_threads)
            .build();

        let ctx = sd.create_context().map_err(|e| {
            let err_msg = format!("Fail to create the context. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::InitContext(err_msg)
        })?;

        let ctx = match ctx {
            Context::TextToImage(ctx) => ctx,
            _ => {
                let err_msg = "Fail to get the context for the text-to-image task";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::InitContext(err_msg.into()));
            }
        };

        SD_TEXT_TO_IMAGE.set(Mutex::new(ctx)).map_err(|_| {
            let err_msg = "Failed to initialize the stable diffusion context. Reason: The `SD_TEXT_TO_IMAGE` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "The stable diffusion text-to-image context has been initialized");
    }

    // create the stable diffusion context for the image-to-image task
    if ctx == SDContextType::Full || ctx == SDContextType::ImageToImage {
        let sd = SDBuidler::new_with_standalone_model(Task::ImageToImage, model_file.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_vae_path(vae.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_clip_l_path(clip_l.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_t5xxl_path(t5xxl.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_lora_model_dir(lora_model_dir.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_n_threads(n_threads)
            .build();

        let ctx = sd.create_context().map_err(|e| {
            let err_msg = format!("Fail to create the context. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::InitContext(err_msg)
        })?;

        let ctx = match ctx {
            Context::ImageToImage(ctx) => ctx,
            _ => {
                let err_msg = "Fail to get the context for the image-to-image task";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::InitContext(err_msg.into()));
            }
        };

        SD_IMAGE_TO_IMAGE.set(Mutex::new(ctx)).map_err(|_| {
        let err_msg = "Failed to initialize the stable diffusion context. Reason: The `SD_IMAGE_TO_IMAGE` has already been initialized";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        LlamaCoreError::InitContext(err_msg.into())
    })?;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "The stable diffusion image-to-image context has been initialized");
    }

    Ok(())
}

/// The context to create for the stable diffusion model
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum SDContextType {
    /// `text_to_image` context
    TextToImage,
    /// `image_to_image` context
    ImageToImage,
    /// Both `text_to_image` and `image_to_image` contexts
    Full,
}

/// Initialize the whisper context
pub fn init_whisper_context(
    whisper_metadata: &WhisperMetadata,
    model_file: impl AsRef<Path>,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the audio context");

    // create and initialize the audio context
    let graph = GraphBuilder::new(EngineType::Whisper)?
        .with_config(whisper_metadata.clone())?
        .use_cpu()
        .build_from_files([model_file.as_ref()])?;

    AUDIO_GRAPH.set(Mutex::new(graph)).map_err(|_| {
            let err_msg = "Failed to initialize the audio context. Reason: The `AUDIO_GRAPH` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "The audio context has been initialized");

    Ok(())
}

/// Initialize the piper context
///
/// # Arguments
///
/// * `voice_model` - Path to the voice model file.
///
/// * `voice_config` - Path to the voice config file.
///
/// * `espeak_ng_data` - Path to the espeak-ng data directory.
///
pub fn init_piper_context(
    voice_model: impl AsRef<Path>,
    voice_config: impl AsRef<Path>,
    espeak_ng_data: impl AsRef<Path>,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the piper context");

    let config = serde_json::json!({
        "model": voice_model.as_ref().to_owned(),
        "config": voice_config.as_ref().to_owned(),
        "espeak_data": espeak_ng_data.as_ref().to_owned(),
    });

    // create and initialize the audio context
    let graph = GraphBuilder::new(EngineType::Piper)?
        .use_cpu()
        .build_from_buffer([config.to_string()])?;

    PIPER_GRAPH.set(Mutex::new(graph)).map_err(|_| {
            let err_msg = "Failed to initialize the piper context. Reason: The `PIPER_GRAPH` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "The piper context has been initialized");

    Ok(())
}
