use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlamaCoreError {
    #[error("{0}")]
    Operation(String),
    #[error("{0}")]
    Backend(#[from] BackendError),
}

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("{0}")]
    Compute(String),
    #[error("{0}")]
    ComputeSingle(String),
    #[error("{0}")]
    SetInput(String),
    #[error("{0}")]
    FinishSingle(String),
    #[error("{0}")]
    GetOutputSingle(String),
    #[error("{0}")]
    GetOutput(String),
}
