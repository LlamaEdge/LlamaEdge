pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod models;
pub mod rag;
pub mod utils;

pub use error::LlamaCoreError;

use chat_prompts::PromptTemplateType;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Mutex, RwLock},
};
use utils::{get_output_buffer, set_tensor_data_u8};
use wasmedge_wasi_nn::{
    Error as WasiNnError, Graph as WasiNnGraph, GraphExecutionContext, TensorType,
};

// key: model_name, value: Graph
pub(crate) static CHAT_GRAPHS: OnceCell<Mutex<HashMap<String, Graph>>> = OnceCell::new();
// key: model_name, value: Graph
pub(crate) static EMBEDDING_GRAPHS: OnceCell<Mutex<HashMap<String, Graph>>> = OnceCell::new();
// cache bytes for decoding utf8
pub(crate) static CACHED_UTF8_ENCODINGS: OnceCell<Mutex<Vec<u8>>> = OnceCell::new();
// running mode
pub(crate) static RUNNING_MODE: OnceCell<RwLock<RunningMode>> = OnceCell::new();

pub(crate) const MAX_BUFFER_SIZE: usize = 2usize.pow(14) * 15 + 128;
pub(crate) const OUTPUT_TENSOR: usize = 0;
const PLUGIN_VERSION: usize = 1;

/// Model metadata
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Metadata {
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
    // #[serde(rename = "main-gpu")]
    // pub main_gpu: u64,
    // #[serde(rename = "tensor-split")]
    // pub tensor_split: String,

    // * Context parameters (used by the llama context):
    #[serde(rename = "ctx-size")]
    pub ctx_size: u64,
    #[serde(rename = "batch-size")]
    pub batch_size: u64,

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
}
impl Default for Metadata {
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
            ctx_size: 512,
            batch_size: 512,
            temperature: 1.0,
            top_p: 1.0,
            repeat_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

/// Builder for the `Metadata` struct
#[derive(Debug)]
pub struct MetadataBuilder {
    metadata: Metadata,
}
impl MetadataBuilder {
    pub fn new<S: Into<String>>(model_name: S, model_alias: S, pt: PromptTemplateType) -> Self {
        let metadata = Metadata {
            model_name: model_name.into(),
            model_alias: model_alias.into(),
            prompt_template: pt,
            ..Default::default()
        };

        Self { metadata }
    }

    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.metadata.model_name = name.into();
        self
    }

    pub fn with_model_alias(mut self, alias: impl Into<String>) -> Self {
        self.metadata.model_alias = alias.into();
        self
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

    pub fn build(self) -> Metadata {
        self.metadata
    }
}

/// Wrapper of the `wasmedge_wasi_nn::Graph` struct
#[derive(Debug)]
pub struct Graph {
    pub created: std::time::Duration,
    pub metadata: Metadata,
    _graph: WasiNnGraph,
    context: GraphExecutionContext,
}
impl Graph {
    /// Create a new computation graph from the given metadata.
    pub fn new(metadata: &Metadata) -> Result<Self, String> {
        let config = serde_json::to_string(&metadata).map_err(|e| e.to_string())?;

        // load the model
        let graph = wasmedge_wasi_nn::GraphBuilder::new(
            wasmedge_wasi_nn::GraphEncoding::Ggml,
            wasmedge_wasi_nn::ExecutionTarget::AUTO,
        )
        .config(config)
        .build_from_cache(&metadata.model_alias)
        .map_err(|e| e.to_string())?;

        // initialize the execution context
        let context = graph.init_execution_context().map_err(|e| e.to_string())?;

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?;

        Ok(Self {
            created,
            metadata: metadata.clone(),
            _graph: graph,
            context,
        })
    }

    /// Get the name of the model
    pub fn name(&self) -> &str {
        &self.metadata.model_name
    }

    /// Get the alias of the model
    pub fn alias(&self) -> &str {
        &self.metadata.model_alias
    }

    /// Get the prompt template type
    pub fn prompt_template(&self) -> PromptTemplateType {
        self.metadata.prompt_template
    }

    /// Update metadata
    pub fn update_metadata(&mut self) -> Result<(), LlamaCoreError> {
        // update metadata
        let config = match serde_json::to_string(&self.metadata) {
            Ok(config) => config,
            Err(e) => {
                return Err(LlamaCoreError::Operation(format!(
                    "Fail to serialize metadata to a JSON string. {}",
                    e
                )));
            }
        };
        set_tensor_data_u8(self, 1, config.as_bytes())
    }

    /// Set input uses the data, not only [u8](https://doc.rust-lang.org/nightly/std/primitive.u8.html), but also [f32](https://doc.rust-lang.org/nightly/std/primitive.f32.html), [i32](https://doc.rust-lang.org/nightly/std/primitive.i32.html), etc.
    pub fn set_input<T: Sized>(
        &mut self,
        index: usize,
        tensor_type: TensorType,
        dimensions: &[usize],
        data: impl AsRef<[T]>,
    ) -> Result<(), WasiNnError> {
        self.context.set_input(index, tensor_type, dimensions, data)
    }

    /// Compute the inference on the given inputs.
    pub fn compute(&mut self) -> Result<(), WasiNnError> {
        self.context.compute()
    }

    /// Compute the inference on the given inputs.
    ///
    /// Note that this method is used for the stream mode. It generates one token at a time.
    pub fn compute_single(&mut self) -> Result<(), WasiNnError> {
        self.context.compute_single()
    }

    /// Copy output tensor to out_buffer, return the output’s **size in bytes**.
    pub fn get_output<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T],
    ) -> Result<usize, WasiNnError> {
        self.context.get_output(index, out_buffer)
    }

    /// Copy output tensor to out_buffer, return the output’s **size in bytes**.
    ///
    /// Note that this method is used for the stream mode. It returns one token at a time.
    pub fn get_output_single<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T],
    ) -> Result<usize, WasiNnError> {
        self.context.get_output_single(index, out_buffer)
    }

    /// Clear the computation context.
    ///
    /// Note that this method is used for the stream mode. It clears the context after the stream mode is finished.
    pub fn finish_single(&mut self) -> Result<(), WasiNnError> {
        self.context.fini_single()
    }
}

pub fn init_core_context(
    metadata_for_chats: Option<&[Metadata]>,
    metadata_for_embeddings: Option<&[Metadata]>,
) -> Result<(), LlamaCoreError> {
    if metadata_for_chats.is_none() && metadata_for_embeddings.is_none() {
        return Err(LlamaCoreError::InitContext(
            "Failed to initialize the core context. Please set metadata for chat completions and/or embeddings.".into(),
        ));
    }

    let mut mode = RunningMode::Embeddings;

    if let Some(metadata_chats) = metadata_for_chats {
        let mut chat_graphs = HashMap::new();
        for metadata in metadata_chats {
            let graph = Graph::new(metadata).map_err(|e| {
                LlamaCoreError::InitContext(format!("Failed to create a chat graph. {}", e))
            })?;

            chat_graphs.insert(graph.name().to_string(), graph);
        }
        CHAT_GRAPHS.set(Mutex::new(chat_graphs)).map_err(|_| {
            LlamaCoreError::InitContext(
                "Failed to initialize the core context. Reason: the `CHAT_GRAPHS` has already been initialized".to_string(),
            )
        })?;

        mode = RunningMode::Chat
    }

    if let Some(metadata_embeddings) = metadata_for_embeddings {
        let mut embedding_graphs = HashMap::new();
        for metadata in metadata_embeddings {
            let graph = Graph::new(metadata).map_err(|e| {
                LlamaCoreError::InitContext(format!(
                    "Failed to create a embedding graph. Reason: {}",
                    e
                ))
            })?;

            embedding_graphs.insert(graph.name().to_string(), graph);
        }
        EMBEDDING_GRAPHS
            .set(Mutex::new(embedding_graphs))
            .map_err(|_| {
                LlamaCoreError::InitContext(
                    "Failed to initialize the core context. Reason: The `EMBEDDING_GRAPHS` has already been initialized".to_string(),
                )
            })?;

        if mode == RunningMode::Chat {
            mode = RunningMode::ChatEmbedding;
        }
    }

    RUNNING_MODE.set(RwLock::new(mode)).map_err(|_| {
        LlamaCoreError::InitContext("Failed to initialize the core context. Reason: The `RUNNING_MODE` has already been initialized".to_string())
    })?;

    Ok(())
}

/// Initialize the core context for RAG scenarios.
pub fn init_rag_core_context(
    metadata_for_chats: &[Metadata],
    metadata_for_embeddings: &[Metadata],
) -> Result<(), LlamaCoreError> {
    // chat models
    if metadata_for_chats.is_empty() {
        return Err(LlamaCoreError::InitContext(
            "The metadata for chat models is empty".to_string(),
        ));
    }
    let mut chat_graphs = HashMap::new();
    for metadata in metadata_for_chats {
        let graph = Graph::new(metadata).map_err(|e| {
            LlamaCoreError::InitContext(format!("Failed to create a chat graph. {}", e))
        })?;

        chat_graphs.insert(graph.name().to_string(), graph);
    }
    CHAT_GRAPHS.set(Mutex::new(chat_graphs)).map_err(|_| {
        LlamaCoreError::InitContext("The `CHAT_GRAPHS` has already been initialized".to_string())
    })?;

    // embedding models
    if metadata_for_embeddings.is_empty() {
        return Err(LlamaCoreError::InitContext(
            "The metadata for embeddings is empty".to_string(),
        ));
    }
    let mut embedding_graphs = HashMap::new();
    for metadata in metadata_for_embeddings {
        let graph = Graph::new(metadata).map_err(|e| {
            LlamaCoreError::InitContext(format!(
                "Failed to create a embedding graph. Reason: {}",
                e
            ))
        })?;

        embedding_graphs.insert(graph.name().to_string(), graph);
    }
    EMBEDDING_GRAPHS
        .set(Mutex::new(embedding_graphs))
        .map_err(|_| {
            LlamaCoreError::InitContext(
                "The `EMBEDDING_GRAPHS` has already been initialized".to_string(),
            )
        })?;

    // set running mode
    RUNNING_MODE.set(RwLock::new(RunningMode::Rag)).map_err(|_| {
            LlamaCoreError::InitContext("Failed to initialize the core context. Reason: The `RUNNING_MODE` has already been initialized".to_string())
        })?;

    Ok(())
}

/// Get the plugin info
///
/// Note that it is required to call `init_core_context` before calling this function.
pub fn get_plugin_info() -> Result<PluginInfo, LlamaCoreError> {
    match running_mode()? {
        RunningMode::Embeddings => {
            let embedding_graphs =
                EMBEDDING_GRAPHS
                    .get()
                    .ok_or(LlamaCoreError::Operation(String::from(
                        "Fail to get the underlying value of `EMBEDDING_GRAPHS`.",
                    )))?;
            let embedding_graphs = embedding_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}",
                    e
                ))
            })?;

            let graph = embedding_graphs
                .values()
                .next()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `GRAPH`.",
                )))?;

            get_plugin_info_by_graph(graph)
        }
        _ => {
            let chat_graphs = CHAT_GRAPHS
                .get()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `CHAT_GRAPHS`.",
                )))?;
            let chat_graphs = chat_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `CHAT_GRAPHS`. {}",
                    e
                ))
            })?;

            let graph = chat_graphs
                .values()
                .next()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `GRAPH`.",
                )))?;

            get_plugin_info_by_graph(graph)
        }
    }
}

fn get_plugin_info_by_graph(graph: &Graph) -> Result<PluginInfo, LlamaCoreError> {
    // get the plugin metadata
    let output_buffer = get_output_buffer(graph, PLUGIN_VERSION)?;
    let metadata: serde_json::Value = serde_json::from_slice(&output_buffer[..]).map_err(|e| {
        LlamaCoreError::Operation(format!(
            "Fail to deserialize the plugin metadata. {msg}",
            msg = e
        ))
    })?;

    // get build number of the plugin
    let plugin_build_number =
        metadata["llama_build_number"]
            .as_u64()
            .ok_or(LlamaCoreError::Operation(String::from(
                "Failed to convert the build number of the plugin to u64",
            )))?;

    // get commit id of the plugin
    let plugin_commit = metadata["llama_commit"]
        .as_str()
        .ok_or(LlamaCoreError::Operation(String::from(
            "Failed to convert the commit id of the plugin to string",
        )))?;

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
    let mode = RUNNING_MODE
        .get()
        .ok_or(LlamaCoreError::Operation(String::from(
            "Fail to get the underlying value of `RUNNING_MODE`.",
        )))?
        .read()
        .map_err(|e| {
            LlamaCoreError::Operation(format!("Fail to acquire the lock of `RUNNING_MODE`. {}", e))
        })?
        .to_owned();

    Ok(mode.to_owned())
}
