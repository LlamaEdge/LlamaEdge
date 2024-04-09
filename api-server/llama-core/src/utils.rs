//! Define utility functions.

use crate::{error::LlamaCoreError, CHAT_GRAPHS, EMBEDDING_GRAPHS};

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
    separator.push('\n');
    println!("{}", separator);
    total_len
}

pub(crate) fn print_log_end_separator(ch: Option<&str>, len: Option<usize>) {
    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(len.unwrap_or(100)).as_str());
    separator.push_str("\n\n");
    println!("{}", separator);
}

pub(crate) fn gen_chat_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

/// Return the names of the chat models.
pub fn chat_model_names() -> Result<Vec<String>, LlamaCoreError> {
    let chat_graphs = CHAT_GRAPHS
        .get()
        .ok_or(LlamaCoreError::Operation(String::from(
            "Fail to get the underlying value of `CHAT_GRAPHS`.",
        )))?;

    let chat_graphs = chat_graphs.lock().map_err(|e| {
        LlamaCoreError::Operation(format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e))
    })?;

    let mut model_names = Vec::new();
    for model_name in chat_graphs.keys() {
        model_names.push(model_name.clone());
    }

    Ok(model_names)
}

/// Return the names of the embedding models.
pub fn embedding_model_names() -> Result<Vec<String>, LlamaCoreError> {
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

    let mut model_names = Vec::new();
    for model_name in embedding_graphs.keys() {
        model_names.push(model_name.clone());
    }

    Ok(model_names)
}
