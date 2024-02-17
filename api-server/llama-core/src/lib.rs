pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod models;

pub use error::LlamaCoreError;

use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use wasi_nn::{Error as WasiNnError, Graph as WasiNnGraph, GraphExecutionContext, TensorType};

use crate::error::BackendError;

pub(crate) static MAX_BUFFER_SIZE: OnceCell<usize> = OnceCell::new();
pub(crate) static CTX_SIZE: OnceCell<usize> = OnceCell::new();
pub(crate) static GRAPH: OnceCell<Mutex<Graph>> = OnceCell::new();
pub(crate) static METADATA: OnceCell<Metadata> = OnceCell::new();
pub static UTF8_ENCODINGS: OnceCell<Mutex<Vec<u8>>> = OnceCell::new();

pub(crate) const MAX_BUFFER_SIZE_EMBEDDING: usize = 4096 * 15 + 128;

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct Metadata {
    #[serde(rename = "enable-log")]
    pub log_enable: bool,
    #[serde(rename = "ctx-size")]
    pub ctx_size: u64,
    #[serde(rename = "n-predict")]
    pub n_predict: u64,
    #[serde(rename = "n-gpu-layers")]
    pub n_gpu_layers: u64,
    #[serde(rename = "batch-size")]
    pub batch_size: u64,
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
    #[serde(rename = "embedding")]
    pub embeddings: bool,
    #[serde(skip_serializing_if = "Option::is_none", rename = "reverse-prompt")]
    pub reverse_prompt: Option<String>,
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
        let graph = wasi_nn::GraphBuilder::new(
            wasi_nn::GraphEncoding::Ggml,
            wasi_nn::ExecutionTarget::AUTO,
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

pub(crate) fn print_log_begin_separator(
    title: impl AsRef<str>,
    ch: Option<&str>,
    len: Option<usize>,
) -> usize {
    let title = format!(" [LOG: {}] ", title.as_ref());

    let total_len: usize = len.unwrap_or(100);
    let separator_len: usize = (total_len - title.len()) / 2;

    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push_str(&title);
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push_str("\n");
    println!("{}", separator);
    total_len
}

pub(crate) fn print_log_end_separator(ch: Option<&str>, len: Option<usize>) {
    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(len.unwrap_or(100)).as_str());
    separator.push_str("\n");
    println!("{}", separator);
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
        .map_err(|e| LlamaCoreError::InitContext(e))?;

    GRAPH.set(Mutex::new(graph)).map_err(|_| {
        LlamaCoreError::InitContext(format!("The `GRAPH` has already been initialized"))
    })?;

    // set `CTX_SIZE`
    CTX_SIZE.set(metadata.ctx_size as usize).map_err(|e| {
        LlamaCoreError::InitContext(format!(
            "The `CTX_SIZE` has already been initialized: {}",
            e
        ))
    })?;

    // set `MAX_BUFFER_SIZE`
    MAX_BUFFER_SIZE
        .set(metadata.ctx_size as usize * 6)
        .map_err(|e| {
            LlamaCoreError::InitContext(format!(
                "The `MAX_BUFFER_SIZE` has already been initialized: {}",
                e
            ))
        })?;

    // set `METADATA`
    METADATA.set(metadata.clone()).map_err(|_| {
        LlamaCoreError::InitContext(format!("The `METADATA` has already been initialized"))
    })?;

    Ok(())
}

/// Get the plugin info
///
/// Note that it is required to call `init_core_context` before calling this function.
pub fn get_plugin_info() -> Result<PluginInfo, LlamaCoreError> {
    let graph = get_graph()?;

    // get the plugin metadata
    let max_output_size = get_max_buffer_size()?;
    let mut output_buffer = vec![0u8; max_output_size];
    let mut output_size: usize = graph.get_output(1, &mut output_buffer).map_err(|e| {
        LlamaCoreError::Backend(BackendError::GetOutput(format!(
            "Fail to get plugin metadata. {msg}",
            msg = e.to_string()
        )))
    })?;
    output_size = std::cmp::min(max_output_size, output_size);
    let metadata: serde_json::Value = serde_json::from_slice(&output_buffer[..output_size])
        .map_err(|e| {
            LlamaCoreError::Operation(format!(
                "Fail to deserialize the plugin metadata. {msg}",
                msg = e.to_string()
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

pub fn debug_core() {
    let ctx_size = CTX_SIZE.get().unwrap();
    println!("[CORE] CTX_SIZE: {:?}", ctx_size);

    let max_buffer_size = MAX_BUFFER_SIZE.get().unwrap();
    println!("[CORE] MAX_BUFFER_SIZE: {:?}", max_buffer_size);

    let metadata = METADATA.get().unwrap();
    println!("[CORE] METADATA: {:?}", metadata);
}

pub(crate) fn get_graph() -> Result<std::sync::MutexGuard<'static, Graph>, LlamaCoreError> {
    let graph = GRAPH.get().ok_or(LlamaCoreError::Operation(String::from(
        "Fail to get the underlying value of `GRAPH`.",
    )))?;

    let graph = graph.lock().map_err(|e| {
        LlamaCoreError::Operation(format!(
            "Fail to acquire the lock of `GRAPH`. {}",
            e.to_string()
        ))
    })?;

    Ok(graph)
}

pub(crate) fn get_max_buffer_size() -> Result<usize, LlamaCoreError> {
    // Retrieve the output.
    let max_buffer_size = MAX_BUFFER_SIZE
        .get()
        .ok_or(LlamaCoreError::Operation(String::from(
            "Fail to get the underlying value of `MAX_BUFFER_SIZE`.",
        )))?;

    Ok(*max_buffer_size)
}

/// Version info of the `wasi-nn_ggml` plugin, including the build number and the commit id.
pub struct PluginInfo {
    pub build_number: u64,
    pub commit_id: String,
}
