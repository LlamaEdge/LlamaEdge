pub mod chat;
pub mod completions;
pub mod error;
pub mod models;

pub use error::LlamaCoreError;

use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf, str::FromStr, sync::Mutex};
use wasi_nn::{Error as WasiNnError, Graph as WasiNnGraph, GraphExecutionContext, TensorType};

pub static MAX_BUFFER_SIZE: OnceCell<usize> = OnceCell::new();
pub static CTX_SIZE: OnceCell<usize> = OnceCell::new();
pub(crate) static GRAPH: OnceCell<Mutex<Graph>> = OnceCell::new();
pub static METADATA: OnceCell<Metadata> = OnceCell::new();
pub static UTF8_ENCODINGS: OnceCell<Mutex<Vec<u8>>> = OnceCell::new();

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct Metadata {
    #[serde(rename = "enable-log")]
    log_enable: bool,
    #[serde(rename = "ctx-size")]
    ctx_size: u64,
    #[serde(rename = "n-predict")]
    n_predict: u64,
    #[serde(rename = "n-gpu-layers")]
    n_gpu_layers: u64,
    #[serde(rename = "batch-size")]
    batch_size: u64,
    #[serde(rename = "temp")]
    temperature: f64,
    #[serde(rename = "top-p")]
    top_p: f64,
    #[serde(rename = "repeat-penalty")]
    repeat_penalty: f64,
    #[serde(rename = "presence-penalty")]
    presence_penalty: f64,
    #[serde(rename = "frequency-penalty")]
    frequency_penalty: f64,
    #[serde(skip_serializing_if = "Option::is_none", rename = "reverse-prompt")]
    reverse_prompt: Option<String>,
}

#[derive(Debug)]
pub(crate) struct Graph {
    _graph: WasiNnGraph,
    context: GraphExecutionContext,
}
impl Graph {
    pub fn new(model_alias: impl AsRef<str>, options: &Metadata) -> Result<Self, String> {
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

        Ok(Self {
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ModelInfo {
    name: String,
}
impl ModelInfo {
    fn new(name: impl AsRef<str>) -> Self {
        Self {
            name: name.as_ref().to_string(),
        }
    }
}
