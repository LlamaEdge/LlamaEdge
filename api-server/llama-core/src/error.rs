//! Error types for the Llama Core library.

use thiserror::Error;

/// Error types for the Llama Core library.
#[derive(Error, Debug)]
pub enum LlamaCoreError {
    /// Errors in General operation.
    #[error("{0}")]
    Operation(String),
    /// Errors in Context initialization.
    #[error("Failed to initialize RAG context. {0}")]
    InitContext(String),
    /// Errors thrown by the wasi-nn-ggml plugin and runtime.
    #[error("{0}")]
    Backend(#[from] BackendError),
}

/// Error types for wasi-nn errors.
#[derive(Error, Debug)]
pub enum BackendError {
    /// Errors in setting the input tensor.
    #[error("{0}")]
    SetInput(String),
    /// Errors in the model inference.
    #[error("{0}")]
    Compute(String),
    /// Errors in the model inference in the stream mode.
    #[error("{0}")]
    ComputeSingle(String),
    /// Errors in getting the output tensor.
    #[error("{0}")]
    GetOutput(String),
    /// Errors in getting the output tensor in the stream mode.
    #[error("{0}")]
    GetOutputSingle(String),
    /// Errors in cleaning up the computation context in the stream mode.
    #[error("{0}")]
    FinishSingle(String),
}
