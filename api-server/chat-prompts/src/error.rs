use endpoints::chat::ChatCompletionRole;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, PromptError>;

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum PromptError {
    #[error("No messages to create prompt from.")]
    NoMessages,
    #[error("No user message to create prompt from.")]
    NoUserMessage,
    #[error("Unknown chat completion role: {0:?}")]
    UnknownRole(ChatCompletionRole),
    #[error("Unknown prompt template type: {0}")]
    UnknownPromptTemplateType(String),
}
