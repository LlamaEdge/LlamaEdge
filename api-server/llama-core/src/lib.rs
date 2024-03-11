pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod models;
pub mod rag;
pub(crate) mod utils;

pub use error::LlamaCoreError;

use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use wasmedge_wasi_nn::{
    Error as WasiNnError, Graph as WasiNnGraph, GraphExecutionContext, TensorType,
};

use crate::error::BackendError;

pub(crate) static CTX_SIZE: OnceCell<usize> = OnceCell::new();
pub(crate) static GRAPH: OnceCell<Mutex<Graph>> = OnceCell::new();
pub(crate) static METADATA: OnceCell<Metadata> = OnceCell::new();
pub static UTF8_ENCODINGS: OnceCell<Mutex<Vec<u8>>> = OnceCell::new();

pub(crate) const MAX_BUFFER_SIZE: usize = 2usize.pow(14) * 15 + 128;
pub(crate) const MAX_BUFFER_SIZE_EMBEDDING: usize = 2usize.pow(14) * 15 + 128;

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct Metadata {
    // * Plugin parameters (used by this plugin):
    #[serde(rename = "enable-log")]
    pub log_enable: bool,
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub log_prompts: bool,
    // #[serde(rename = "enable-debug-log")]
    // pub debug_log: bool,
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

#[derive(Debug)]
pub struct Graph {
    pub name: String,
    pub created: std::time::Duration,
    _graph: WasiNnGraph,
    context: GraphExecutionContext,
}
impl Graph {
    pub fn new(
        model_name: impl Into<String>,
        model_alias: impl AsRef<str>,
        options: &Metadata,
    ) -> Result<Self, String> {
        let config = serde_json::to_string(&options).map_err(|e| e.to_string())?;

        // load the model
        let graph = wasmedge_wasi_nn::GraphBuilder::new(
            wasmedge_wasi_nn::GraphEncoding::Ggml,
            wasmedge_wasi_nn::ExecutionTarget::AUTO,
        )
        .config(config)
        .build_from_cache(model_alias.as_ref())
        .map_err(|e| e.to_string())?;

        // initialize the execution context
        let context = graph.init_execution_context().map_err(|e| e.to_string())?;

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?;

        Ok(Self {
            name: model_name.into(),
            created,
            _graph: graph,
            context,
        })
    }

    pub fn set_input<T: Sized>(
        &mut self,
        index: usize,
        tensor_type: TensorType,
        dimensions: &[usize],
        data: impl AsRef<[T]>,
    ) -> Result<(), WasiNnError> {
        self.context.set_input(index, tensor_type, dimensions, data)
    }

    pub fn compute(&mut self) -> Result<(), WasiNnError> {
        self.context.compute()
    }

    pub fn compute_single(&mut self) -> Result<(), WasiNnError> {
        self.context.compute_single()
    }

    pub fn get_output<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T],
    ) -> Result<usize, WasiNnError> {
        self.context.get_output(index, out_buffer)
    }

    pub fn get_output_single<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T],
    ) -> Result<usize, WasiNnError> {
        self.context.get_output_single(index, out_buffer)
    }

    pub fn finish_single(&mut self) -> Result<(), WasiNnError> {
        self.context.fini_single()
    }
}

/// Initialize the core context
///
/// # Arguments
///
/// * `metadata` - The metadata of the model
///
/// * `model_name` - The name of the model
///
/// * `model_alias` - The alias of the model
///
pub fn init_core_context(
    metadata: &Metadata,
    model_name: impl AsRef<str>,
    model_alias: impl AsRef<str>,
) -> Result<(), LlamaCoreError> {
    let graph = Graph::new(model_name.as_ref(), model_alias.as_ref(), metadata)
        .map_err(LlamaCoreError::InitContext)?;

    GRAPH.set(Mutex::new(graph)).map_err(|_| {
        LlamaCoreError::InitContext("The `GRAPH` has already been initialized".to_string())
    })?;

    // set `CTX_SIZE`
    CTX_SIZE.set(metadata.ctx_size as usize).map_err(|e| {
        LlamaCoreError::InitContext(format!(
            "The `CTX_SIZE` has already been initialized: {}",
            e
        ))
    })?;

    // set `METADATA`
    METADATA.set(metadata.clone()).map_err(|_| {
        LlamaCoreError::InitContext("The `METADATA` has already been initialized".to_string())
    })?;

    Ok(())
}

/// Get the plugin info
///
/// Note that it is required to call `init_core_context` before calling this function.
pub fn get_plugin_info() -> Result<PluginInfo, LlamaCoreError> {
    let graph = get_graph()?;

    // get the plugin metadata
    let mut output_buffer = vec![0u8; MAX_BUFFER_SIZE];
    let mut output_size: usize = graph.get_output(1, &mut output_buffer).map_err(|e| {
        LlamaCoreError::Backend(BackendError::GetOutput(format!(
            "Fail to get plugin metadata. {msg}",
            msg = e
        )))
    })?;
    output_size = std::cmp::min(MAX_BUFFER_SIZE, output_size);
    let metadata: serde_json::Value = serde_json::from_slice(&output_buffer[..output_size])
        .map_err(|e| {
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

pub(crate) fn get_graph() -> Result<std::sync::MutexGuard<'static, Graph>, LlamaCoreError> {
    let graph = GRAPH.get().ok_or(LlamaCoreError::Operation(String::from(
        "Fail to get the underlying value of `GRAPH`.",
    )))?;

    let graph = graph.lock().map_err(|e| {
        LlamaCoreError::Operation(format!("Fail to acquire the lock of `GRAPH`. {}", e))
    })?;

    Ok(graph)
}

/// Version info of the `wasi-nn_ggml` plugin, including the build number and the commit id.
pub struct PluginInfo {
    pub build_number: u64,
    pub commit_id: String,
}
