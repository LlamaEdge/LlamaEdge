use thiserror::Error;

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ChatError {
    #[error("Context full")]
    ContextFull(String),
    #[error("The prompt is too long")]
    PromptTooLong,
    #[error("Fail to compute")]
    Operation(String),
}
